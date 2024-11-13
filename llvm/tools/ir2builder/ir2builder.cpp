//===-- ir2builder.cpp - Transpiler from IR to builder API ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a pass, which converts LLVM IR into llvm::IRBuilder
// API in textual form.
//
// This tool can be used to simplify IR construction using IRBuilder by
// writing the IR by hand and then converting it using this pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <sstream>

using namespace llvm;

cl::OptionCategory Ir2BCat("ir2builder Options");

/** LLVM IR to convert */
static cl::opt<std::string>
    InputFilename(cl::Positional, cl::desc("<input .ll file>"), cl::init("-"));

/**
 * Output file for the generated C++ code.
 * If not set then stdout will be used.
 */
static cl::opt<std::string> OutputFilename("o",
                                           cl::desc("Override output filename"),
                                           cl::value_desc("filename"),
                                           cl::cat(Ir2BCat));

/** Set this when you don't use `using namespace llvm` (don't forget :: at the
 * end) */
static cl::opt<std::string> ScopePrefix(
    "scope-prefix",
    cl::desc(
        "All generated calls to LLVM API will be prefixed with this scope. The "
        "scope has to end with '::' (e.g.: '-scope-prefix=llvm::')"),
    cl::cat(Ir2BCat), cl::init(""));

/** Set this to your variable name for IRBuilder instance */
static cl::opt<std::string> BuilderName(
    "builder-name",
    cl::desc("IRBuilder variable name that will be used in generated code"),
    cl::cat(Ir2BCat), cl::init("Builder"));

/** Set this to your LLVMContext variable name */
static cl::opt<std::string> ContextName(
    "context-name",
    cl::desc("Context variable name that will be used in generated code"),
    cl::cat(Ir2BCat), cl::init("Ctx"));

/** Set this to your llvm::Module * name */
static cl::opt<std::string> ModuleName(
    "module-name",
    cl::desc("Module variable name that will be used in generated code"),
    cl::cat(Ir2BCat), cl::init("Mod"));

/** Set this if you want custom data layout */
static cl::opt<std::string> ClDataLayout("data-layout",
                                         cl::desc("data layout string to use"),
                                         cl::value_desc("layout-string"),
                                         cl::init(""), cl::cat(Ir2BCat));

/**
 * This will generate fully compilable C++ program with main, which will
 * use generated calls to create the original LLVM IR.
 * Main purpose of this is to use it for testing and verification that
 * generated program is correct and compilable.
 * For example you can generate the code, run this program and then use
 * llvm-diff to see if it matches the original:
 *     ```
 *     ./ir2builder code.ll --runnable > main.cpp &&
 *     g++ main.cpp `./llvm-conf --cxxflags --ldflags --system-libs --libs core`
 * -I /llvm/include/ -o main &&
 *     ./main > generated.ll &&
 *     ./llvm-diff code.ll generated.ll
 *     ```
 */
static cl::opt<bool> GenerateRunnable(
    "runnable", cl::desc("Generates whole cpp compilable program with main"),
    cl::init(false), cl::cat(Ir2BCat));

/**
 * Disables verification of loaded llvm IR.
 * Keep in mind that this will most likely result in C++ error as it probably
 * won't be possible to create Builder calls for this.
 */
static cl::opt<bool>
    DisableVerify("disable-verify", cl::Hidden,
                  cl::desc("Do not run verifier on input LLVM (dangerous!)"),
                  cl::cat(Ir2BCat));

/**
 * Sets the order of traversal for the LLVM IR.
 * When enabled the traversal will be in reverse post order, which can handle
 * when values are defined after (text-wise) their use.
 * On the other hand using just linear traversal will also include parts that
 * are outside of the graph (dead blocks).
 */
static cl::opt<bool>
    UseRPO("use-rpo",
           cl::desc("Traverses IR in reverse post order. This can help with "
                    "\"was not declared\" errors"),
           cl::init(true), cl::cat(Ir2BCat));

/**
 * @brief Transpiler from LLVM IR into IRBuilder API calls
 * The main purpose for this class is to hold variable counter and variable
 * names for needed resources, such as LLVMContext.
 */
class IR2Builder {
private:
  unsigned long varI = 0;
  std::string llvmPrefix;
  std::string builder;
  std::string ctx;
  std::string modName;

  std::vector<std::string> phiIncomings;

  inline bool hasName(const Value *op) {
    return !isa<Constant>(op) && !isa<InlineAsm>(op);
  }

  void outputAttr(Attribute att, raw_ostream &OS);

  std::string getNextVar();
  std::string getLinkage(GlobalValue &gv);
  std::string getThreadLocalMode(GlobalValue::ThreadLocalMode tlm);
  std::string getCmpPredicate(CmpInst::Predicate p);
  std::string getAtomicRMWOp(AtomicRMWInst::BinOp op);
  std::string getAtomicOrdering(AtomicOrdering ao);
  std::string getSyncScopeID(SyncScope::ID sys);
  std::string getCallingConv(CallingConv::ID cc);
  std::string getConstantRange(ConstantRange &cr);

  std::string getVal(const Value *op);
  std::string getConst(const Constant *op);
  std::string getType(const Type *t);
  std::string getInlineAsm(const InlineAsm *op);
  std::string getMetadata(const Metadata *op);

public:
  IR2Builder()
      : llvmPrefix(ScopePrefix), builder(BuilderName), ctx(ContextName),
        modName(ModuleName) {}

  /**
   * Calls convert for all the functions in passed in module
   * @param M Module to call convert over
   * @param OS Stream to which output the builder calls
   */
  void convert(Module &M, raw_ostream &OS);

  /**
   * Converts a function into IRBuilder API calls
   * @param F Function to convert
   * @param OS Stream to which output the builder calls
   */
  void convert(Function &F, raw_ostream &OS);

  /**
   * Converts an instruction into IRBuilder API calls
   * @param I Instruction to convert
   * @param OS Stream to which output the builder calls
   * @note Unsupported instructions or their operands should result
   *       in a TODO comment.
   */
  void convert(const Instruction *I, raw_ostream &OS);
};

std::string IR2Builder::getNextVar() {
  return "v0" + std::to_string(varI++);
}

static std::string to_str(bool b) { return b ? "true" : "false"; }

static std::string escape(std::string str) {
  std::stringstream ss;
  for (unsigned char C : str) {
    if (C == '\\')
      ss << '\\' << C;
    else if (isPrint(C) && C != '"')
      ss << C;
    else
      ss << '\\' << hexdigit(C >> 4) << hexdigit(C & 0x0F);
  }
  return ss.str();
}

static std::string sanitize(std::string s) {
  std::stringstream ss;
  for (size_t i = 0; i < s.size(); ++i) {
    if (!std::isalnum(s[i])) {
      ss << "_" << static_cast<unsigned>(s[i]) << "_";
    } else {
      ss << s[i];
    }
  }
  return ss.str();
}

std::string IR2Builder::getLinkage(llvm::GlobalValue &gv) {
  std::string link = llvmPrefix + "GlobalValue::";

  if (gv.hasExternalLinkage()) {
    return link + "ExternalLinkage";
  }
  if (gv.hasAvailableExternallyLinkage()) {
    return link + "AvailableExternallyLinkage";
  }
  if (gv.hasLinkOnceAnyLinkage()) {
    return link + "LinkOnceAnyLinkage";
  }
  if (gv.hasLinkOnceODRLinkage()) {
    return link + "LinkOnceODRLinkage";
  }
  if (gv.hasWeakAnyLinkage()) {
    return link + "WeakAnyLinkage";
  }
  if (gv.hasWeakODRLinkage()) {
    return link + "WeakODRLinkage";
  }
  if (gv.hasAppendingLinkage()) {
    return link + "AppendingLinkage";
  }
  if (gv.hasInternalLinkage()) {
    return link + "InternalLinkage";
  }
  if (gv.hasPrivateLinkage()) {
    return link + "PrivateLinkage";
  }
  if (gv.hasExternalWeakLinkage()) {
    return link + "ExternalWeakLinkage";
  }
  if (gv.hasCommonLinkage()) {
    return link + "CommonLinkage";
  }

  return "/* Unknown linkage */";
}

std::string IR2Builder::getAtomicRMWOp(AtomicRMWInst::BinOp op) {
  switch (op) {
  case AtomicRMWInst::BinOp::Xchg:
    return llvmPrefix + "AtomicRMWInst::BinOp::Xchg";
  case AtomicRMWInst::BinOp::Add:
    return llvmPrefix + "AtomicRMWInst::BinOp::Add";
  case AtomicRMWInst::BinOp::Sub:
    return llvmPrefix + "AtomicRMWInst::BinOp::Sub";
  case AtomicRMWInst::BinOp::And:
    return llvmPrefix + "AtomicRMWInst::BinOp::And";
  case AtomicRMWInst::BinOp::Nand:
    return llvmPrefix + "AtomicRMWInst::BinOp::Nand";
  case AtomicRMWInst::BinOp::Or:
    return llvmPrefix + "AtomicRMWInst::BinOp::Or";
  case AtomicRMWInst::BinOp::Xor:
    return llvmPrefix + "AtomicRMWInst::BinOp::Xor";
  case AtomicRMWInst::BinOp::Max:
    return llvmPrefix + "AtomicRMWInst::BinOp::Max";
  case AtomicRMWInst::BinOp::Min:
    return llvmPrefix + "AtomicRMWInst::BinOp::Min";
  case AtomicRMWInst::BinOp::UMax:
    return llvmPrefix + "AtomicRMWInst::BinOp::UMax";
  case AtomicRMWInst::BinOp::UMin:
    return llvmPrefix + "AtomicRMWInst::BinOp::UMin";
  case AtomicRMWInst::BinOp::FAdd:
    return llvmPrefix + "AtomicRMWInst::BinOp::FAdd";
  case AtomicRMWInst::BinOp::FSub:
    return llvmPrefix + "AtomicRMWInst::BinOp::FSub";
  case AtomicRMWInst::BinOp::FMax:
    return llvmPrefix + "AtomicRMWInst::BinOp::FMax";
  case AtomicRMWInst::BinOp::FMin:
    return llvmPrefix + "AtomicRMWInst::BinOp::FMin";
  case AtomicRMWInst::BinOp::UIncWrap:
    return llvmPrefix + "AtomicRMWInst::BinOp::UIncWrap";
  case AtomicRMWInst::BinOp::UDecWrap:
    return llvmPrefix + "AtomicRMWInst::BinOp::UDecWrap";
  default:
    return "/* TODO: Unknown AtomicRMW operator (using value) */ " +
           std::to_string(op);
  }
}

std::string IR2Builder::getAtomicOrdering(AtomicOrdering ao) {
  switch (ao) {
  case AtomicOrdering::NotAtomic:
    return llvmPrefix + "AtomicOrdering::NotAtomic";
  case AtomicOrdering::Unordered:
    return llvmPrefix + "AtomicOrdering::Unordered";
  case AtomicOrdering::Monotonic:
    return llvmPrefix + "AtomicOrdering::Monotonic";
  case AtomicOrdering::Acquire:
    return llvmPrefix + "AtomicOrdering::Acquire";
  case AtomicOrdering::Release:
    return llvmPrefix + "AtomicOrdering::Release";
  case AtomicOrdering::AcquireRelease:
    return llvmPrefix + "AtomicOrdering::AcquireRelease";
  case AtomicOrdering::SequentiallyConsistent:
    return llvmPrefix + "AtomicOrdering::SequentiallyConsistent";
  default:
    return "/* TODO: Unknown atomic ordering */";
  }
}

std::string IR2Builder::getCmpPredicate(CmpInst::Predicate p) {
  switch (p) {
  case CmpInst::Predicate::FCMP_FALSE:
    return llvmPrefix + "CmpInst::Predicate::FCMP_FALSE";
  case CmpInst::Predicate::FCMP_OEQ:
    return llvmPrefix + "CmpInst::Predicate::FCMP_OEQ";
  case CmpInst::Predicate::FCMP_OGT:
    return llvmPrefix + "CmpInst::Predicate::FCMP_OGT";
  case CmpInst::Predicate::FCMP_OGE:
    return llvmPrefix + "CmpInst::Predicate::FCMP_OGE";
  case CmpInst::Predicate::FCMP_OLT:
    return llvmPrefix + "CmpInst::Predicate::FCMP_OLT";
  case CmpInst::Predicate::FCMP_OLE:
    return llvmPrefix + "CmpInst::Predicate::FCMP_OLE";
  case CmpInst::Predicate::FCMP_ONE:
    return llvmPrefix + "CmpInst::Predicate::FCMP_ONE";
  case CmpInst::Predicate::FCMP_ORD:
    return llvmPrefix + "CmpInst::Predicate::FCMP_ORD";
  case CmpInst::Predicate::FCMP_UNO:
    return llvmPrefix + "CmpInst::Predicate::FCMP_UNO";
  case CmpInst::Predicate::FCMP_UEQ:
    return llvmPrefix + "CmpInst::Predicate::FCMP_UEQ";
  case CmpInst::Predicate::FCMP_UGT:
    return llvmPrefix + "CmpInst::Predicate::FCMP_UGT";
  case CmpInst::Predicate::FCMP_UGE:
    return llvmPrefix + "CmpInst::Predicate::FCMP_UGE";
  case CmpInst::Predicate::FCMP_ULT:
    return llvmPrefix + "CmpInst::Predicate::FCMP_ULT";
  case CmpInst::Predicate::FCMP_ULE:
    return llvmPrefix + "CmpInst::Predicate::FCMP_ULE";
  case CmpInst::Predicate::FCMP_UNE:
    return llvmPrefix + "CmpInst::Predicate::FCMP_UNE";
  case CmpInst::Predicate::FCMP_TRUE:
    return llvmPrefix + "CmpInst::Predicate::FCMP_TRUE";
  case CmpInst::Predicate::ICMP_EQ:
    return llvmPrefix + "CmpInst::Predicate::ICMP_EQ";
  case CmpInst::Predicate::ICMP_NE:
    return llvmPrefix + "CmpInst::Predicate::ICMP_NE";
  case CmpInst::Predicate::ICMP_UGT:
    return llvmPrefix + "CmpInst::Predicate::ICMP_UGT";
  case CmpInst::Predicate::ICMP_UGE:
    return llvmPrefix + "CmpInst::Predicate::ICMP_UGE";
  case CmpInst::Predicate::ICMP_ULT:
    return llvmPrefix + "CmpInst::Predicate::ICMP_ULT";
  case CmpInst::Predicate::ICMP_ULE:
    return llvmPrefix + "CmpInst::Predicate::ICMP_ULE";
  case CmpInst::Predicate::ICMP_SGT:
    return llvmPrefix + "CmpInst::Predicate::ICMP_SGT";
  case CmpInst::Predicate::ICMP_SGE:
    return llvmPrefix + "CmpInst::Predicate::ICMP_SGE";
  case CmpInst::Predicate::ICMP_SLT:
    return llvmPrefix + "CmpInst::Predicate::ICMP_SLT";
  case CmpInst::Predicate::ICMP_SLE:
    return llvmPrefix + "CmpInst::Predicate::ICMP_SLE";
  default:
    return "/* TODO: Unknown CMP predicate (using value) */ " +
           std::to_string(p);
  }
}

std::string IR2Builder::getSyncScopeID(SyncScope::ID sys) {
  if (sys == SyncScope::System)
    return llvmPrefix + "SyncScope::System";
  else if (sys == SyncScope::SingleThread)
    return llvmPrefix + "SyncScope::SingleThread";
  else
    return "/* TODO: Unknown SyncScope ID (using value) */ " +
           std::to_string(sys);
}

std::string IR2Builder::getThreadLocalMode(GlobalValue::ThreadLocalMode tlm) {
  switch (tlm) {
  case GlobalValue::ThreadLocalMode::NotThreadLocal:
    return llvmPrefix + "GlobalValue::ThreadLocalMode::NotThreadLocal";
  case GlobalValue::ThreadLocalMode::GeneralDynamicTLSModel:
    return llvmPrefix + "GlobalValue::ThreadLocalMode::GeneralDynamicTLSModel";
  case GlobalValue::ThreadLocalMode::LocalDynamicTLSModel:
    return llvmPrefix + "GlobalValue::ThreadLocalMode::LocalDynamicTLSModel";
  case GlobalValue::ThreadLocalMode::InitialExecTLSModel:
    return llvmPrefix + "GlobalValue::ThreadLocalMode::InitialExecTLSModel";
  case GlobalValue::ThreadLocalMode::LocalExecTLSModel:
    return llvmPrefix + "GlobalValue::ThreadLocalMode::LocalExecTLSModel";
  default:
    return "/* TODO: Unknown ThreadLocalMode (using value) */ " +
           std::to_string(tlm);
  }
}

std::string IR2Builder::getCallingConv(CallingConv::ID cc) {
  switch (cc) {
  case CallingConv::C:
    return llvmPrefix + "CallingConv::C";
  case CallingConv::Fast:
    return llvmPrefix + "CallingConv::Fast";
  case CallingConv::Cold:
    return llvmPrefix + "CallingConv::Cold";
  case CallingConv::GHC:
    return llvmPrefix + "CallingConv::GHC";
  case CallingConv::HiPE:
    return llvmPrefix + "CallingConv::HiPE";
  case CallingConv::AnyReg:
    return llvmPrefix + "CallingConv::AnyReg";
  case CallingConv::PreserveMost:
    return llvmPrefix + "CallingConv::PreserveMost";
  case CallingConv::PreserveAll:
    return llvmPrefix + "CallingConv::PreserveAll";
  case CallingConv::Swift:
    return llvmPrefix + "CallingConv::Swift";
  case CallingConv::CXX_FAST_TLS:
    return llvmPrefix + "CallingConv::CXX_FAST_TLS";
  case CallingConv::Tail:
    return llvmPrefix + "CallingConv::Tail";
  case CallingConv::CFGuard_Check:
    return llvmPrefix + "CallingConv::CFGuard_Check";
  case CallingConv::SwiftTail:
    return llvmPrefix + "CallingConv::SwiftTail";
  case CallingConv::PreserveNone:
    return llvmPrefix + "CallingConv::PreserveNone";
  case CallingConv::FirstTargetCC:
    return llvmPrefix + "CallingConv::FirstTargetCC";
  // CallingConv::X86_StdCall is the same as FirstTargetCC
  case CallingConv::X86_FastCall:
    return llvmPrefix + "CallingConv::X86_FastCall";
  case CallingConv::ARM_APCS:
    return llvmPrefix + "CallingConv::ARM_APCS";
  case CallingConv::ARM_AAPCS:
    return llvmPrefix + "CallingConv::ARM_AAPCS";
  case CallingConv::ARM_AAPCS_VFP:
    return llvmPrefix + "CallingConv::ARM_AAPCS_VFP";
  case CallingConv::MSP430_INTR:
    return llvmPrefix + "CallingConv::MSP430_INTR";
  case CallingConv::X86_ThisCall:
    return llvmPrefix + "CallingConv::X86_ThisCall";
  case CallingConv::PTX_Kernel:
    return llvmPrefix + "CallingConv::PTX_Kernel";
  case CallingConv::PTX_Device:
    return llvmPrefix + "CallingConv::PTX_Device";
  case CallingConv::SPIR_FUNC:
    return llvmPrefix + "CallingConv::SPIR_FUNC";
  case CallingConv::SPIR_KERNEL:
    return llvmPrefix + "CallingConv::SPIR_KERNEL";
  case CallingConv::Intel_OCL_BI:
    return llvmPrefix + "CallingConv::Intel_OCL_BI";
  case CallingConv::X86_64_SysV:
    return llvmPrefix + "CallingConv::X86_64_SysV";
  case CallingConv::Win64:
    return llvmPrefix + "CallingConv::Win64";
  case CallingConv::X86_VectorCall:
    return llvmPrefix + "CallingConv::X86_VectorCall";
  case CallingConv::DUMMY_HHVM:
    return llvmPrefix + "CallingConv::DUMMY_HHVM";
  case CallingConv::DUMMY_HHVM_C:
    return llvmPrefix + "CallingConv::DUMMY_HHVM_C";
  case CallingConv::X86_INTR:
    return llvmPrefix + "CallingConv::X86_INTR";
  case CallingConv::AVR_INTR:
    return llvmPrefix + "CallingConv::AVR_INTR";
  case CallingConv::AVR_SIGNAL:
    return llvmPrefix + "CallingConv::AVR_SIGNAL";
  case CallingConv::AVR_BUILTIN:
    return llvmPrefix + "CallingConv::AVR_BUILTIN";
  case CallingConv::AMDGPU_VS:
    return llvmPrefix + "CallingConv::AMDGPU_VS";
  case CallingConv::AMDGPU_GS:
    return llvmPrefix + "CallingConv::AMDGPU_GS";
  case CallingConv::AMDGPU_PS:
    return llvmPrefix + "CallingConv::AMDGPU_PS";
  case CallingConv::AMDGPU_CS:
    return llvmPrefix + "CallingConv::AMDGPU_CS";
  case CallingConv::AMDGPU_KERNEL:
    return llvmPrefix + "CallingConv::AMDGPU_KERNEL";
  case CallingConv::X86_RegCall:
    return llvmPrefix + "CallingConv::X86_RegCall";
  case CallingConv::AMDGPU_HS:
    return llvmPrefix + "CallingConv::AMDGPU_HS";
  case CallingConv::MSP430_BUILTIN:
    return llvmPrefix + "CallingConv::MSP430_BUILTIN";
  case CallingConv::AMDGPU_LS:
    return llvmPrefix + "CallingConv::AMDGPU_LS";
  case CallingConv::AMDGPU_ES:
    return llvmPrefix + "CallingConv::AMDGPU_ES";
  case CallingConv::AArch64_VectorCall:
    return llvmPrefix + "CallingConv::AArch64_VectorCall";
  case CallingConv::AArch64_SVE_VectorCall:
    return llvmPrefix + "CallingConv::AArch64_SVE_VectorCall";
  case CallingConv::WASM_EmscriptenInvoke:
    return llvmPrefix + "CallingConv::WASM_EmscriptenInvoke";
  case CallingConv::AMDGPU_Gfx:
    return llvmPrefix + "CallingConv::AMDGPU_Gfx";
  case CallingConv::M68k_INTR:
    return llvmPrefix + "CallingConv::M68k_INTR";
  case CallingConv::AArch64_SME_ABI_Support_Routines_PreserveMost_From_X0:
    return llvmPrefix +
           "CallingConv::AArch64_SME_ABI_Support_Routines_PreserveMost_From_X0";
  case CallingConv::AArch64_SME_ABI_Support_Routines_PreserveMost_From_X2:
    return llvmPrefix +
           "CallingConv::AArch64_SME_ABI_Support_Routines_PreserveMost_From_X2";
  case CallingConv::AMDGPU_CS_Chain:
    return llvmPrefix + "CallingConv::AMDGPU_CS_Chain";
  case CallingConv::AMDGPU_CS_ChainPreserve:
    return llvmPrefix + "CallingConv::AMDGPU_CS_ChainPreserve";
  case CallingConv::M68k_RTD:
    return llvmPrefix + "CallingConv::M68k_RTD";
  case CallingConv::GRAAL:
    return llvmPrefix + "CallingConv::GRAAL";
  case CallingConv::ARM64EC_Thunk_X64:
    return llvmPrefix + "CallingConv::ARM64EC_Thunk_X64";
  case CallingConv::ARM64EC_Thunk_Native:
    return llvmPrefix + "CallingConv::ARM64EC_Thunk_Native";
  default:
    return "/* Custom CC */" + std::to_string(cc);
  }
}

std::string IR2Builder::getType(const Type *t) {
  std::string tcall;
  if (auto tt = dyn_cast<IntegerType>(t)) {
    switch (tt->getBitWidth()) {
    case 8:
      tcall = "getInt8Ty()";
      break;
    case 16:
      tcall = "getInt16Ty()";
      break;
    case 32:
      tcall = "getInt32Ty()";
      break;
    case 64:
      tcall = "getInt64Ty()";
      break;
    default:
      tcall = "getIntNTy(" + std::to_string(tt->getBitWidth()) + ")";
      break;
    }
  } else if (t->isVoidTy())
    tcall = "getVoidTy()";
  else if (t->isFloatTy())
    tcall = "getFloatTy()";
  else if (t->isDoubleTy())
    tcall = "getDoubleTy()";
  else if (auto st = dyn_cast<StructType>(t)) {
    std::string elements = llvmPrefix + "ArrayRef<Type *>(";
    if (st->getNumElements() > 1)
      elements += "{";
    bool first = true;
    for (auto e : st->elements()) {
      if (!first)
        elements += ", ";
      elements += getType(e);
      first = false;
    }
    if (st->getNumElements() > 1)
      elements += "}";
    elements += ")";
    return llvmPrefix + "StructType::create(" + ctx + ", " + elements + ")";
  } else if (auto at = dyn_cast<ArrayType>(t)) {
    return llvmPrefix + "ArrayType::get(" + getType(at->getElementType()) +
           ", " + std::to_string(at->getArrayNumElements()) + ")";
  } else if (t->isBFloatTy())
    tcall = "getBFloatTy()";
  else if (auto vt = dyn_cast<VectorType>(t)) {
    std::string elemCount =
        llvmPrefix + "ElementCount::get(" +
        std::to_string(vt->getElementCount().getKnownMinValue()) + ", " +
        to_str(vt->getElementCount().isScalable()) + ")";
    return llvmPrefix + "VectorType::get(" + getType(vt->getElementType()) +
           ", " + elemCount + ")";
  } else if (auto ft = dyn_cast<FunctionType>(t)) {
    tcall = llvmPrefix + "FunctionType::get(" + getType(ft->getReturnType()) +
            ", {";

    bool isFirst = true;
    for (auto a : ft->params()) {
      if (!isFirst)
        tcall += ", ";
      tcall += getType(a);
      isFirst = false;
    }
    tcall += "}, " + to_str(ft->isVarArg()) + ")";
    return tcall;
  } else if (t->isPointerTy())
    tcall = "getPtrTy()";
  else if (t->isHalfTy()) {
    return llvmPrefix + "Type::getHalfTy(" + ctx + ")";
  } else if (t->isBFloatTy()) {
    return llvmPrefix + "Type::getBFloatTy(" + ctx + ")";
  } else if (t->isX86_FP80Ty()) {
    return llvmPrefix + "Type::getX86_FP80Ty(" + ctx + ")";
  } else if (t->isFP128Ty()) {
    return llvmPrefix + "Type::getFP128Ty(" + ctx + ")";
  } else if (t->isPPC_FP128Ty()) {
    return llvmPrefix + "Type::getPPC_FP128Ty(" + ctx + ")";
  } else if (t->isX86_AMXTy()) {
    return llvmPrefix + "Type::getX86_AMXTy(" + ctx + ")";
  } else if (t->isLabelTy()) {
    return llvmPrefix + "Type::getLabelTy(" + ctx + ")";
  } else if (t->isMetadataTy()) {
    return llvmPrefix + "Type::getMetadataTy(" + ctx + ")";
  } else if (t->isTokenTy()) {
    return llvmPrefix + "Type::getTokenTy(" + ctx + ")";
  } else {
    return "/* TODO: Unknown type */";
  }

  return builder + "." + tcall;
}

std::string IR2Builder::getConst(const Constant *c) {
  if (auto ci = dyn_cast<ConstantInt>(c)) {
    // TODO: Sign has to be determined
    auto cval = ci->getValue();
    return llvmPrefix + "ConstantInt::get(" + getType(c->getType()) + ", " +
           std::to_string(cval.getSExtValue()) + ")";
  } else if (auto cf = dyn_cast<ConstantFP>(c)) {
    auto cval = cf->getValue();
    double dval = cval.convertToDouble();
    std::string val = std::to_string(dval);
    if (std::isnan(dval) || std::isinf(dval))
      val = "\"" + val + "\"";
    // TODO: Handle double to string conversion to include all digits
    return llvmPrefix + "ConstantFP::get(" + getType(c->getType()) + ", " +
           val + ")";
  } else if (auto at = dyn_cast<ConstantAggregate>(c)) {
    std::string values;
    bool first = true;
    for (unsigned i = 0; i < at->getNumOperands(); ++i) {
      if (!first)
        values += ", ";
      values += getVal(at->getOperand(i));
      first = false;
    }

    std::string className;
    if (isa<ConstantArray>(c)) {
      className = "ConstantArray";
      values = llvmPrefix + "ArrayRef<" + llvmPrefix + "Constant *>(" +
               (at->getNumOperands() > 1 ? std::string("{") : std::string("")) +
               values +
               (at->getNumOperands() > 1 ? std::string("}") : std::string("")) +
               ")";
    } else if (isa<ConstantStruct>(c))
      className = "ConstantStruct";
    else if (isa<ConstantVector>(c)) {
      values = "{" + values + "}";
      // ConstantVector does not take type as 1st arg
      return llvmPrefix + "ConstantVector::get(" + values + ")";
    } else
      return "/* TODO: Unknown aggregate constant */";

    return llvmPrefix + className + "::get(" + getType(c->getType()) + ", " +
           values + ")";
  } else if (auto cds = dyn_cast<ConstantDataSequential>(c)) {
    std::string values;
    std::string className;
    std::string elemTy = "/* TODO */";
    if (isa<ConstantDataArray>(c))
      className = "ConstantDataArray";
    else if (isa<ConstantDataVector>(c))
      className = "ConstantDataVector";
    else
      return "/* TODO: Unknown data sequential constant */";
    if (cds->isString()) {
      values = "";
      bool first = true;
      for (auto a : cds->getAsString().str()) {
        if (first) {
          values += std::to_string(static_cast<uint8_t>(a));
          first = false;
        } else {
          values += ", " + std::to_string(static_cast<uint8_t>(a));
        }
      }
      return llvmPrefix + className + "::get(" + ctx + ", " + llvmPrefix +
             "ArrayRef<uint8_t>({" + values + "}))";
    } else if (cds->isCString()) {
      values = "";
      bool first = true;
      for (auto a : cds->getAsCString().str()) {
        if (first) {
          values += std::to_string(static_cast<uint8_t>(a));
          first = false;
        } else {
          values += ", " + std::to_string(static_cast<uint8_t>(a));
        }
      }
      return llvmPrefix + className + "::get(" + ctx + ", " + llvmPrefix +
             "ArrayRef<uint8_t>({" + values + "}))";
    } else {
      Type *elemT = cds->getElementType();
      if (elemT->isIntegerTy()) {
        // There can be only 8, 16, 32 or 64 ints in ConstantDataVector
        elemTy = "uint" + std::to_string(elemT->getIntegerBitWidth()) + "_t";
      } else if (elemT->isDoubleTy()) {
        elemTy = "double";
      } else if (elemT->isFloatTy()) {
        elemTy = "float";
      }
      values = llvmPrefix + "ArrayRef<" + elemTy + ">(";
      if (cds->getNumElements() > 1)
        values += "{";
      bool first = true;
      for (unsigned i = 0; i < cds->getNumElements(); ++i) {
        if (!first)
          values += ", ";
        if (elemT->isIntegerTy()) {
          values += std::to_string(cds->getElementAsInteger(i));
        } else if (elemT->isDoubleTy()) {
          values += std::to_string(cds->getElementAsDouble(i));
        } else if (elemT->isFloatTy()) {
          values += std::to_string(cds->getElementAsFloat(i));
        } else
          return "/* Unknown type in data sequential constant */";
        first = false;
      }
      if (cds->getNumElements() > 1)
        values += "}";
      values += ")";
    }

    return llvmPrefix + className + "::get(" + ctx + ", " + values + ")";
  } else if (isa<ConstantAggregateZero>(c)) {
    return llvmPrefix + "ConstantAggregateZero::get(" + getType(c->getType()) +
           ")";
  } else if (isa<PoisonValue>(c)) {
    return llvmPrefix + "PoisonValue::get(" + getType(c->getType()) + ")";
  } else if (isa<UndefValue>(c)) {
    return llvmPrefix + "UndefValue::get(" + getType(c->getType()) + ")";
  } else if (auto ba = dyn_cast<BlockAddress>(c)) {
    return llvmPrefix + "BlockAddress::get(" + getVal(ba->getFunction()) +
           ", " + getVal(ba->getBasicBlock()) + ")";
  } else if (isa<ConstantPointerNull>(c)) {
    return llvmPrefix + "ConstantPointerNull::get(" + getType(c->getType()) +
           ")";
  } else if (auto ctn = dyn_cast<ConstantTargetNone>(c)) {
    auto tetType = ctn->getType();

    std::string typeStr = "{";
    bool first = true;
    for (unsigned i = 0; i < tetType->getNumTypeParameters(); ++i) {
      if (!first)
        typeStr += ", ";
      typeStr += getType(tetType->getTypeParameter(i));
      first = false;
    }
    typeStr += "}";

    std::string intsStr = "{";
    first = true;
    for (unsigned i = 0; i < tetType->getNumIntParameters(); ++i) {
      if (!first)
        intsStr += ", ";
      intsStr += std::to_string(tetType->getIntParameter(i));
      first = false;
    }
    intsStr += "}";

    std::string tetName = "\"" + escape(tetType->getName().str()) + "\"";
    std::string tet = llvmPrefix + "TargetExtType::get(" + ctx + ", " +
                      tetName + ", " + typeStr + ", " + intsStr + ")";

    return llvmPrefix + "ConstantTargetNone::get(" + tet + ")";
  } else if (isa<ConstantTokenNone>(c)) {
    return llvmPrefix + "ConstantTokenNone::get(" + ctx + ")";
  } else if (auto ce = dyn_cast<ConstantExpr>(c)) {
    (void)ce;
    return "/* TODO: ConstantExpr creation */";
    // TODO: Dunno how to create this... Fails either on out of range or
    // even on incorrect opcode
    // return llvmPrefix + "ConstantExpr::get(" +
    // std::to_string(ce->getOpcode()) + ", " +
    //       getVal(ce->getOperand(0)) + ", " + getVal(ce->getOperand(1)) + ")";
  } else if (auto ce = dyn_cast<ConstantPtrAuth>(c)) {
    (void)ce;
    return "/* TODO: ConstantPtrAuth value creation */";
  } else if (auto ce = dyn_cast<DSOLocalEquivalent>(c)) {
    (void)ce;
    return "/* TODO: DSOLocalEquivalent value creation */";
  } else if (auto ce = dyn_cast<NoCFIValue>(c)) {
    (void)ce;
    return "/* TODO: NoCFIValue value creation */";
  } else if (auto ce = dyn_cast<GlobalValue>(c)) {
    // This should not really happen as getVal should be always called
    return getVal(ce);
  }

  return "/* TODO: Constant creation */";
}

std::string IR2Builder::getInlineAsm(const InlineAsm *op) {
  auto getAsmDialect = [llvmPrefix = llvmPrefix](InlineAsm::AsmDialect d) {
    switch (d) {
    case InlineAsm::AsmDialect::AD_ATT:
      return llvmPrefix + "InlineAsm::AsmDialect::AD_ATT";
    case InlineAsm::AsmDialect::AD_Intel:
      return llvmPrefix + "InlineAsm::AsmDialect::AD_Intel";
    default:
      return "/* TODO: Unknown AsmDialect (using value) */" + std::to_string(d);
    }
  };

  return llvmPrefix + "InlineAsm::get(" + getType(op->getFunctionType()) +
         ", " + "\"" + escape(op->getAsmString()) + "\", " + "\"" +
         escape(op->getConstraintString()) + "\", " +
         to_str(op->hasSideEffects()) + ", " + to_str(op->isAlignStack()) +
         ", " + getAsmDialect(op->getDialect()) + ", " +
         to_str(op->canThrow()) + ")";
}

std::string IR2Builder::getMetadata(const Metadata *op) {
  if (auto mdn = dyn_cast<MDNode>(op)) {
    std::string args = "{";
    bool first = true;
    for (unsigned i = 0; i < mdn->getNumOperands(); ++i) {
      if (!first)
        args += ", ";
      args += getMetadata(mdn->getOperand(i));
      first = false;
    }
    args += "}";
    return llvmPrefix + "MDNode::get(" + ctx + ", " + args + ")";
  } else if (auto vam = dyn_cast<ValueAsMetadata>(op)) {
    return llvmPrefix + "ValueAsMetadata::get(" + getVal(vam->getValue()) + ")";
  } else if (auto mds = dyn_cast<MDString>(op)) {
    return llvmPrefix + "MDString::get(" + ctx + ", \"" +
           escape(mds->getString().str()) + "\")";
  } else {
    return "/* TODO: Metadata creation */";
  }
}

// This is a DEBUG function in Value, so lets copy it here for NDEBUG as well
static std::string getNameOrAsOperand(const Value *v) {
  if (!v->getName().empty())
    return std::string(v->getName());

  std::string BBName;
  raw_string_ostream OS(BBName);
  v->printAsOperand(OS, false);
  return OS.str();
}

std::string IR2Builder::getVal(const Value *op) {
  if (!op)
    return "nullptr";
  if (isa<Constant>(op) && !isa<GlobalValue>(op)) {
    return getConst(dyn_cast<Constant>(op));
  } else if (auto ina = dyn_cast<InlineAsm>(op)) {
    return getInlineAsm(ina);
  } else if (auto mtd = dyn_cast<MetadataAsValue>(op)) {
    return llvmPrefix + "MetadataAsValue::get(" + ctx + ", " +
           getMetadata(mtd->getMetadata()) + ")";
  }
  std::string opName = getNameOrAsOperand(op);
  if (opName[0] == '%')
    opName.erase(opName.begin());
  std::string pref = "v_";
  if (isa<GlobalValue>(op))
    pref = "g_";
  return pref + sanitize(opName);
}

std::string getBinArithOp(std::string name, std::string op1, std::string op2,
                          const Instruction *I) {
  return "Create" + name + "(" + op1 + ", " + op2 + ", \"\", " +
         to_str(I->hasNoUnsignedWrap()) + ", " + to_str(I->hasNoSignedWrap()) +
         ")";
}

std::string getFPBinArithOp(std::string name, std::string op1, std::string op2,
                            const Instruction *I) {
  return "Create" + name + "(" + op1 + ", " + op2 + ")";
  // TODO: Handle FPMathTag
}

std::string IR2Builder::getConstantRange(ConstantRange &cr) {
  std::stringstream ss;
  unsigned NumWords = divideCeil(cr.getBitWidth(), 64);
  auto numWordsStr = std::to_string(NumWords);
  auto numBitsStr = std::to_string(cr.getBitWidth());
  auto lower = std::to_string(cr.getLower().getLimitedValue());
  auto upper = std::to_string(cr.getUpper().getLimitedValue());
  ss << llvmPrefix << "ConstantRange(APInt(" << numBitsStr << ", " << lower
     << ", true), "
     << "APInt(" << numBitsStr << ", " << upper << ", true))";
  return ss.str();
}

void IR2Builder::outputAttr(Attribute att, raw_ostream &OS) {
  // TODO: Handle special cases detected using "has" methods
  // see Attribute::getAsString(bool InAttrGrp)
  if (att.isStringAttribute()) {
    OS << llvmPrefix << "Attribute::get(" << ctx << ", \""
       << att.getKindAsString() << "\"";
    auto val = att.getValueAsString();
    if (val.empty()) {
      OS << ")";
    } else {
      OS << ", \"";
      printEscapedString(val, OS);
      OS << "\")";
    }
  } else if (att.isIntAttribute()) {
    OS << llvmPrefix << "Attribute::get(" << ctx << ", (" << llvmPrefix
       << "Attribute::AttrKind)" << std::to_string(att.getKindAsEnum())
       << ", static_cast<uint64_t>(" << att.getValueAsInt() << "))";
  } else if (att.isEnumAttribute()) {
    OS << llvmPrefix << "Attribute::get(" << ctx << ", \""
       << att.getNameFromAttrKind(att.getKindAsEnum()) << "\")";
  } else if (att.isTypeAttribute()) {
    OS << llvmPrefix << "Attribute::get(" << ctx << ", (" << llvmPrefix
       << "Attribute::AttrKind)" << std::to_string(att.getKindAsEnum()) << ", "
       << getType(att.getValueAsType()) << ")";
  } else if (att.isConstantRangeAttribute()) {
    auto cr = att.getValueAsConstantRange();
    OS << llvmPrefix << "Attribute::get(" << ctx << ", (" << llvmPrefix
       << "Attribute::AttrKind)" << std::to_string(att.getKindAsEnum()) << ", "
       << getConstantRange(cr) << ")";
  } else if (att.isConstantRangeListAttribute()) {
    std::string args = "{";
    bool first = true;
    for (auto cr : att.getValueAsConstantRangeList()) {
      if (!first)
        args += ", ";
      args += getConstantRange(cr);
      first = false;
    }
    args += "}";
    OS << llvmPrefix << "Attribute::get(" << ctx << ", (" << llvmPrefix
       << "Attribute::AttrKind)" << std::to_string(att.getKindAsEnum()) << ", "
       << llvmPrefix << args << ")";
  } else {
    OS << "/* TODO: Attribute creation */";
  }
}

void IR2Builder::convert(Module &M, raw_ostream &OS) {
  // Prologue
  if (GenerateRunnable) {
    // Top comment
    OS << "// This file was autogenerated using ir2builder tool\n";

    // Includes
    // Currently we include all possibly needed files as this is done
    // before the conversion of functions
    OS << R"(
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include <cstdint>

using namespace llvm;

int main(int argc, char** argv) {
    InitLLVM X(argc, argv);
)";

    // Used objects and variables
    OS << "    LLVMContext " << ctx << ";\n"
       << "    Module *" << modName
       << "= new Module(\"ir2builder generated module\", " << ctx << ");\n"
       << "    IRBuilder<> " << builder << "(" << ctx << ");\n";

    // Global  generation
    for (auto &G : M.globals()) {
      std::string initVal = "nullptr";
      if (G.hasInitializer()) {
        // In case of BlockAddress initializer we need to declare the function
        // and basic block so it can be used
        if (auto ba = dyn_cast<BlockAddress>(G.getInitializer())) {
          OS << "auto " << getVal(ba->getFunction()) << " = " << modName
             << "->getOrInsertFunction(\"" << ba->getFunction()->getName()
             << "\", " << getType(ba->getFunction()->getFunctionType())
             << ");\n";
          // Extract basic block
          // TODO: The function might need to be created in here
          // since the basic blocks don't exist yet
          OS << "// TODO: Basic block extraction\n";
        }
        initVal = getVal(G.getInitializer());
      }

      std::string gname = getVal(&G);
      OS << "auto " << gname << " = new " << llvmPrefix << "GlobalVariable(*"
         << modName << ", " << getType(G.getValueType()) << ", "
         << to_str(G.isConstant()) << ", " << getLinkage(G) << ", " << initVal
         << ", \"" << escape(G.getName().str()) << "\", "
         << "nullptr"
         << ", " << getThreadLocalMode(G.getThreadLocalMode()) << ", "
         << std::to_string(G.getAddressSpace()) << ", "
         << to_str(G.isExternallyInitialized()) << ");\n";
    }
  }

  // Function generation
  for (Function &F : M) {
    if (!F.isDeclaration())
      convert(F, OS);
  }

  // Epilogue
  if (GenerateRunnable) {
    // Print generated code
    OS << "    " << modName << "->print(outs(), nullptr);\n";
    // Cleanup
    OS << "    delete " << modName << ";\n"
       << "    return 0;\n}\n";
  }
}

void IR2Builder::convert(Function &F, raw_ostream &OS) {
  phiIncomings.clear();
  // Function
  OS << "{\n\n";
  auto fDecl = getVal(&F);
  OS << llvmPrefix << "Function *" << fDecl;
  OS << " = " << llvmPrefix << "Function::Create("
     << getType(F.getFunctionType()) << ", " << getLinkage(F) << ", \""
     << F.getName() << "\", " << modName << ");\n";

  OS << "\n";

  // Set attributes
  if (F.getCallingConv() != CallingConv::C) { // C is default
    OS << fDecl << "->setCallingConv(" + getCallingConv(F.getCallingConv())
       << ");\n";
  }
  // TODO: Handle attributes with values
  if (F.getAttributes().hasFnAttrs()) {
    for (auto att : F.getAttributes().getFnAttrs()) {
      OS << fDecl << "->addFnAttr(";
      outputAttr(att, OS);
      OS << ");\n";
    }
  }
  if (F.getAttributes().hasRetAttrs()) {
    for (auto att : F.getAttributes().getRetAttrs()) {
      OS << fDecl << "->addRetAttr(";
      outputAttr(att, OS);
      OS << ");\n";
    }
  }
  for (size_t i = 0; i < F.arg_size(); ++i) {
    if (F.getAttributes().hasParamAttrs(i)) {
      for (auto att : F.getAttributes().getAttributes(i + 1)) {
        OS << fDecl << "->addParamAttr(" << i << ", ";
        outputAttr(att, OS);
        OS << ");\n";
      }
    }
  }

  // Save arguments into variables for easy access
  for (unsigned i = 0; i < F.arg_size(); ++i) {
    OS << "auto " << getVal(F.getArg(i)) << " = " << fDecl << "->getArg("
       << std::to_string(i) << ");\n";
  }

  if (UseRPO) {
    ReversePostOrderTraversal<Function *> RPOT(&F);

    // Basic block declaration in order
    for (BasicBlock *BB : RPOT) {
      std::string bbName = getVal(BB);
      OS << llvmPrefix << "BasicBlock* " << bbName << " = " << llvmPrefix
         << "BasicBlock::Create(" << ctx << ", \"" << BB->getName() << "\", "
         << fDecl << ");\n";
    }

    OS << "\n";

    for (auto *BB : RPOT) {
      OS << builder << "."
         << "SetInsertPoint(" << getVal(BB) << ");\n";

      for (auto J = BB->begin(), JE = BB->end(); J != JE; ++J) {
        const Instruction *Inst = &*J;
        convert(Inst, OS);
      }

      OS << "\n";
    }
  } else {
    for (BasicBlock &BB : F) {
      std::string bbName = getVal(&BB);
      OS << llvmPrefix << "BasicBlock* " << bbName << " = " << llvmPrefix
         << "BasicBlock::Create(" << ctx << ", \"" << BB.getName() << "\", "
         << fDecl << ");\n";
    }

    OS << "\n";

    for (auto &BB : F) {
      OS << builder << "."
         << "SetInsertPoint(" << getVal(&BB) << ");\n";

      for (auto J = BB.begin(), JE = BB.end(); J != JE; ++J) {
        const Instruction *Inst = &*J;
        convert(Inst, OS);
      }

      OS << "\n";
    }
  }

  // Output incoming values assignment into phis, this is needed as they
  // might refer to a value not yet defined in the time of phi definition
  for (auto l : phiIncomings) {
    OS << l;
  }

  OS << "}\n";
}

void IR2Builder::convert(const Instruction *I, raw_ostream &OS) {
  std::string call;

  std::string op1 = "/* TODO */";
  std::string op2 = "/* TODO */";
  std::string op3 = "/* TODO */";

  if (I->getNumOperands() > 0) {
    op1 = getVal(I->getOperand(0));
  }
  if (I->getNumOperands() > 1) {
    op2 = getVal(I->getOperand(1));
  }
  if (I->getNumOperands() > 2) {
    op3 = getVal(I->getOperand(2));
  }

  switch (I->getOpcode()) {
  case Instruction::Ret: {
    if (I->getNumOperands() == 0)
      call = "CreateRetVoid()";
    else
      call = "CreateRet(" + op1 + ")";
  } break;
  case Instruction::Br: {
    const BranchInst *bi = dyn_cast<BranchInst>(I);
    if (bi->isUnconditional()) {
      call = "CreateBr(" + op1 + ")";
    } else {
      call = "CreateCondBr(" + op1 + ", " + op3 + ", " + op2 + ")";
    }
  } break;
  case Instruction::Switch: {
    auto swI = dyn_cast<SwitchInst>(I);
    call = "CreateSwitch(" + op1 + ", " + op2 + ", " +
           std::to_string(swI->getNumCases()) + ")";

    std::string swVar = getNextVar();
    // No need to save temporary var into symTable
    OS << "auto " << swVar << " = " << builder << "." << call << ";\n";

    for (auto c : swI->cases()) {
      OS << swVar << "->addCase(" << getVal(c.getCaseValue()) << ", "
         << getVal(c.getCaseSuccessor()) << ");\n";
    }
    return;
  } break;
  case Instruction::IndirectBr: {
    auto inbrI = dyn_cast<IndirectBrInst>(I);
    call = "CreateIndirectBr(" + op1 + ", " +
           std::to_string(inbrI->getNumDestinations()) + ")";
    std::string inbrVar = getNextVar();
    OS << "auto " << inbrVar << " = " << builder << "." << call << ";\n";

    for (auto c : inbrI->successors()) {
      OS << inbrVar << "->addDestination(" << getVal(c) << ");\n";
    }
  } break;
  case Instruction::Invoke: {
    auto invI = dyn_cast<InvokeInst>(I);
    std::string args = "{";
    bool first = true;
    for (unsigned i = 0; i < invI->arg_size(); ++i) {
      if (!first)
        args += ", ";
      args += getVal(invI->getArgOperand(i));
      first = false;
    }
    args += "}";
    std::string funDecl = getNextVar();
    OS << "auto " << funDecl << " = " << modName << "->getOrInsertFunction(\""
       << I->getOperand(2)->getName() << "\", "
       << getType(invI->getFunctionType()) << ");\n";
    call = "CreateInvoke(" + funDecl + ", " + getVal(invI->getNormalDest()) +
           ", " + getVal(invI->getUnwindDest()) + ", " + args + ")";
    // TODO: Handle operand bundles
  } break;
  case Instruction::Resume: {
    call = "CreateResume(" + op1 + ")";
  } break;
  case Instruction::Unreachable: {
    call = "CreateUnreachable()";
  } break;
  case Instruction::CleanupRet: {
    auto curI = dyn_cast<CleanupReturnInst>(I);
    call =
        "CreateCleanupRet(" + op1 + ", " + getVal(curI->getUnwindDest()) + ")";
  } break;
  case Instruction::CatchRet: {
    call = "CreateCatchRet(" + op1 + ", " + op2 + ")";
  } break;
  case Instruction::CatchSwitch: {
    auto swI = dyn_cast<CatchSwitchInst>(I);
    call = "CreateCatchSwitch(" + op1 + ", " + op2 + ", " +
           std::to_string(swI->getNumHandlers()) + ")";

    std::string swVar = getVal(swI);
    OS << "auto " << swVar << " = " << builder << "." << call << ";\n";
    for (auto c : swI->handlers()) {
      OS << swVar << "->addHandler(" << getVal(c) << ");\n";
    }
    return;
  } break;
  case Instruction::CallBr: {
    auto callbrI = dyn_cast<CallBrInst>(I);
    std::string inddest = "{";
    bool first = true;
    for (unsigned i = 0; i < callbrI->getNumIndirectDests(); ++i) {
      if (!first)
        inddest += ", ";
      inddest += getVal(callbrI->getIndirectDest(i));
      first = false;
    }
    inddest += "}";

    std::string args = "{";
    first = true;
    for (unsigned i = 0; i < callbrI->arg_size(); ++i) {
      if (!first)
        args += ", ";
      args += getVal(callbrI->getArgOperand(i));
      first = false;
    }
    args += "}";
    call = "CreateCallBr(" + getType(callbrI->getFunctionType()) + ", " +
           getVal(I->getOperand(I->getNumOperands() - 1)) + ", " +
           getVal(callbrI->getDefaultDest()) + ", " + inddest + ", " + args +
           ")";
    // TODO: Handle operand bundles
  } break;
  case Instruction::FNeg: {
    call = "CreateFNeg(" + op1 + ")";
    // TODO: Handle FPMathTag
  } break;
  case Instruction::Add: {
    call = getBinArithOp("Add", op1, op2, I);
  } break;
  case Instruction::FAdd: {
    call = getFPBinArithOp("FAdd", op1, op2, I);
  } break;
  case Instruction::Sub: {
    call = getBinArithOp("Sub", op1, op2, I);
  } break;
  case Instruction::FSub: {
    call = getFPBinArithOp("FSub", op1, op2, I);
  } break;
  case Instruction::Mul: {
    call = getBinArithOp("Mul", op1, op2, I);
  } break;
  case Instruction::FMul: {
    call = getFPBinArithOp("FMul", op1, op2, I);
  } break;
  case Instruction::UDiv: {
    call = "CreateUDiv(" + op1 + ", " + op2 + ", \"\", " +
           to_str(I->isExact()) + ")";
  } break;
  case Instruction::SDiv: {
    call = "CreateSDiv(" + op1 + ", " + op2 + ", \"\", " +
           to_str(I->isExact()) + ")";
  } break;
  case Instruction::FDiv: {
    call = getFPBinArithOp("FDiv", op1, op2, I);
  } break;
  case Instruction::URem: {
    call = "CreateURem(" + op1 + ", " + op2 + ")";
  } break;
  case Instruction::SRem: {
    call = "CreateSRem(" + op1 + ", " + op2 + ")";
  } break;
  case Instruction::FRem: {
    call = getFPBinArithOp("FRem", op1, op2, I);
  } break;
  case Instruction::Shl: {
    call = getBinArithOp("Shl", op1, op2, I);
  } break;
  case Instruction::LShr: {
    call = "CreateLShr(" + op1 + ", " + op2 + ", \"\", " +
           to_str(I->isExact()) + ")";
  } break;
  case Instruction::AShr: {
    call = "CreateAShr(" + op1 + ", " + op2 + ", \"\", " +
           to_str(I->isExact()) + ")";
  } break;
  case Instruction::And: {
    call = "CreateAnd(" + op1 + ", " + op2 + ")";
  } break;
  case Instruction::Or: {
    call = "CreateOr(" + op1 + ", " + op2 + ")";
  } break;
  case Instruction::Xor: {
    call = "CreateXor(" + op1 + ", " + op2 + ")";
  } break;
  case Instruction::Alloca: {
    auto alI = dyn_cast<AllocaInst>(I);
    auto val = alI->getArraySize();
    auto valStr = val ? getVal(val) : "nullptr";
    call = "CreateAlloca(" + getType(alI->getAllocatedType()) + ", " +
           std::to_string(alI->getAddressSpace()) + ", " + valStr + ")";
  } break;
  case Instruction::Load: {
    auto lI = dyn_cast<LoadInst>(I);
    call = "CreateLoad(" + getType(I->getType()) + ", " + op1 + ", " +
           to_str(lI->isVolatile()) + ")";
  } break;
  case Instruction::Store: {
    auto sI = dyn_cast<StoreInst>(I);
    call = "CreateStore(" + op1 + ", " + op2 + ", " + to_str(sI->isVolatile()) +
           ")";
  } break;
  case Instruction::GetElementPtr: {
    auto gepI = dyn_cast<GetElementPtrInst>(I);
    std::string strList = llvmPrefix + "ArrayRef<" + llvmPrefix + "Value*>(";
    if (I->getNumOperands() > 2)
      strList += "{";
    bool first = true;
    for (unsigned i = 1; i < I->getNumOperands(); ++i) {
      if (!first)
        strList += ", ";
      strList += getVal(I->getOperand(i));
      first = false;
    }
    if (I->getNumOperands() > 2)
      strList += "}";
    strList += ")";
    // TODO: For some structs the return type is ptr and it fails
    call = "CreateGEP(" + getType(gepI->getResultElementType()) + ", " + op1 +
           ", " + strList + ", \"\", " + to_str(gepI->isInBounds()) + ")";
  } break;
  case Instruction::Fence: {
    auto fI = dyn_cast<FenceInst>(I);
    call = "CreateFence(" + getAtomicOrdering(fI->getOrdering()) + ", " +
           getSyncScopeID(fI->getSyncScopeID()) + ")";
  } break;
  case Instruction::AtomicCmpXchg: {
    auto acmpxI = dyn_cast<AtomicCmpXchgInst>(I);
    call = "CreateAtomicCmpXchg(" + op1 + ", " + op2 + ", " + op3 + ", Align(" +
           std::to_string(acmpxI->getAlign().value()) + "), " +
           getAtomicOrdering(acmpxI->getSuccessOrdering()) + ", " +
           getAtomicOrdering(acmpxI->getFailureOrdering()) + ", " +
           getSyncScopeID(acmpxI->getSyncScopeID()) + ")";
  } break;
  case Instruction::AtomicRMW: {
    auto armwI = dyn_cast<AtomicRMWInst>(I);
    call = "CreateAtomicRMW(" + getAtomicRMWOp(armwI->getOperation()) + ", " +
           op1 + ", " + op2 + ", Align(" +
           std::to_string(armwI->getAlign().value()) + "), " +
           getAtomicOrdering(armwI->getOrdering()) + ", " +
           getSyncScopeID(armwI->getSyncScopeID()) + ")";
  } break;
  case Instruction::Trunc: {
    call = "CreateTrunc(" + op1 + ", " + getType(I->getType()) + ")";
  } break;
  case Instruction::ZExt: {
    call = "CreateZExt(" + op1 + ", " + getType(I->getType()) + ", \"\", " +
           to_str(I->hasNonNeg()) + ")";
  } break;
  case Instruction::SExt: {
    call = "CreateSExt(" + op1 + ", " + getType(I->getType()) + ")";
  } break;
  case Instruction::FPToUI: {
    call = "CreateFPToUI(" + op1 + ", " + getType(I->getType()) + ")";
  } break;
  case Instruction::FPToSI: {
    call = "CreateFPToSI(" + op1 + ", " + getType(I->getType()) + ")";
  } break;
  case Instruction::UIToFP: {
    call = "CreateUIToFP(" + op1 + ", " + getType(I->getType()) + ")";
  } break;
  case Instruction::SIToFP: {
    call = "CreateSIToFP(" + op1 + ", " + getType(I->getType()) + ")";
  } break;
  case Instruction::FPTrunc: {
    call = "CreateFPTrunc(" + op1 + ", " + getType(I->getType()) + ")";
  } break;
  case Instruction::FPExt: {
    call = "CreateFPExt(" + op1 + ", " + getType(I->getType()) + ")";
  } break;
  case Instruction::PtrToInt: {
    call = "CreatePtrToInt(" + op1 + ", " + getType(I->getType()) + ")";
  } break;
  case Instruction::IntToPtr: {
    call = "CreateIntToPtr(" + op1 + ", " + getType(I->getType()) + ")";
  } break;
  case Instruction::BitCast: {
    call = "CreateBitCast(" + op1 + ", " + getType(I->getType()) + ")";
  } break;
  case Instruction::AddrSpaceCast: {
    call = "CreateAddrSpaceCast(" + op1 + ", " + getType(I->getType()) + ")";
  } break;
  case Instruction::CleanupPad: {
    auto cupI = dyn_cast<CleanupPadInst>(I);
    std::string argsStr = "{";
    bool first = true;
    for (unsigned i = 0; i < cupI->arg_size(); ++i) {
      if (!first)
        argsStr += ", ";
      argsStr += getVal(cupI->getArgOperand(i));
      first = false;
    }
    argsStr += "}";
    call = "CreateCleanupPad(" + op1 + ", " + argsStr + ")";
  } break;
  case Instruction::CatchPad: {
    auto capI = dyn_cast<CatchPadInst>(I);
    std::string argsStr = "{";
    bool first = true;
    for (unsigned i = 0; i < capI->arg_size(); ++i) {
      if (!first)
        argsStr += ", ";
      argsStr += getVal(capI->getArgOperand(i));
      first = false;
    }
    argsStr += "}";
    call = "CreateCatchPad(" + op1 + ", " + argsStr + ")";
  } break;
  case Instruction::ICmp: {
    auto cmpI = dyn_cast<CmpInst>(I);
    std::string cmpPredicate =
        llvmPrefix + getCmpPredicate(cmpI->getPredicate());
    call = "CreateICmp(" + cmpPredicate + ", " + op1 + ", " + op2 + ")";
  } break;
  case Instruction::FCmp: {
    auto cmpI = dyn_cast<CmpInst>(I);
    std::string cmpPredicate =
        llvmPrefix + getCmpPredicate(cmpI->getPredicate());
    call = "CreateFCmp(" + cmpPredicate + ", " + op1 + ", " + op2 + ")";
  } break;
  case Instruction::PHI: {
    auto phI = dyn_cast<PHINode>(I);
    call = "CreatePHI(" + getType(I->getType()) + ", " +
           std::to_string(phI->getNumIncomingValues()) + ")";
    std::string phVar = getVal(phI);
    OS << "auto " << phVar << " = " << builder << "." << call << ";\n";

    unsigned i = 0;
    for (auto b : phI->blocks()) {
      auto incVal = getVal(phI->getIncomingValue(i));
      // Phis might contain incoming not yet defined variable in such case
      // we save the line we would output here and output it after the
      // whole function body was outputted
      std::string line =
          phVar + "->addIncoming(" + incVal + ", " + getVal(b) + ");\n";
      phiIncomings.push_back(line);
      ++i;
    }
    return;
  } break;
  case Instruction::Call: {
    auto fcI = dyn_cast<CallBase>(I);
    std::string argDecl = "";
    if (fcI->arg_size() > 0) {
      argDecl = getNextVar();
      OS << "Value *" << argDecl << "[] = {";
      bool isFirst = true;
      for (unsigned i = 0; i < fcI->arg_size(); ++i) {
        if (!isFirst)
          OS << ", ";
        OS << getVal(fcI->getArgOperand(i));
        isFirst = false;
      }
      OS << "};\n";
    }

    /* TODO: Implement
    std::string bundleDecl = "";
    if (fcI->hasOperandBundles()) {
        bundleDecl = getNextVar();
        OS << "Value *" << bundleDecl << "[] = {";
        isFirst = true;
        for(unsigned i = 0; i < fcI->getNumOperandBundles(); ++i) {
            if(!isFirst) OS << ", ";
            // TODO: Probably create OperandBundleUse from the current gotten
    from
            // getOperandBundleAt and cast construct OperandBundleDefT with it,
            // but this requires conversion of Inputs which are Use
            OS << "...";
            isFirst = false;
        }
        OS << "};\n";
    }
    */

    auto fun = dyn_cast<Function>(I->getOperand(I->getNumOperands() - 1));
    std::string funDecl = getNextVar();
    if (!fun) {
      fun = fcI->getCalledFunction();
    }

    if (!fun) {
      assert(fcI->isIndirectCall() && "sanity check");
      if (argDecl.empty()) {
        call = "CreateCall(" + getType(fcI->getFunctionType()) + ", " +
               getVal(I->getOperand(I->getNumOperands() - 1)) + ")";
      } else {
        call = "CreateCall(" + getType(fcI->getFunctionType()) + ", " +
               getVal(I->getOperand(I->getNumOperands() - 1)) + ", " + argDecl +
               ")";
      }
    } else {
      // No need to save this variable as it is a temporary one
      OS << "auto " << funDecl << " = " << modName << "->getOrInsertFunction(\""
         << fun->getName() << "\", " << getType(fun->getFunctionType())
         << ");\n";
      if (!argDecl.empty())
        call = "CreateCall(" + funDecl + ", " + argDecl + ")";
      else
        call = "CreateCall(" + funDecl + ")";
    }
  } break;
  case Instruction::Select: {
    call = "CreateSelect(" + op1 + ", " + op2 + ", " + op3 + ")";
  } break;
  case Instruction::UserOp1: {
    // Internal opcode
    OS << "// TODO: UserOp1 appeared in the IR\n";
    return;
  };
  case Instruction::UserOp2: {
    // Internal opcode
    OS << "// TODO: UserOp2 appeared in the IR\n";
    return;
  };
  case Instruction::VAArg: {
    call = "CreateVAArg(" + op1 + ", " + getType(I->getType()) + ")";
  } break;
  case Instruction::ExtractElement: {
    call = "CreateExtractElement(" + op1 + ", " + op2 + ")";
  } break;
  case Instruction::InsertElement: {
    call = "CreateInsertElement(" + op1 + ", " + op2 + ", " + op3 + ")";
  } break;
  case Instruction::ShuffleVector: {
    auto svI = dyn_cast<ShuffleVectorInst>(I);
    std::string maskStr = "{";
    bool first = true;
    for (int i : svI->getShuffleMask()) {
      if (!first)
        maskStr += ", ";
      maskStr += std::to_string(i);
      first = false;
    }
    maskStr += "}";
    call = "CreateShuffleVector(" + op1 + ", " + op2 + ", " + maskStr + ")";
  } break;
  case Instruction::ExtractValue: {
    auto evI = dyn_cast<ExtractValueInst>(I);
    std::string argStr = "{";
    bool isFirst = true;
    for (auto ind : evI->getIndices()) {
      if (!isFirst)
        argStr += ", ";
      argStr += std::to_string(ind);
      isFirst = false;
    }
    argStr += "}";
    call = "CreateExtractValue(" + op1 + ", " + argStr + ")";
  } break;
  case Instruction::InsertValue: {
    auto ivI = dyn_cast<InsertValueInst>(I);
    std::string argStr = "{";
    bool isFirst = true;
    for (auto ind : ivI->getIndices()) {
      if (!isFirst)
        argStr += ", ";
      argStr += std::to_string(ind);
      isFirst = false;
    }
    argStr += "}";
    call = "CreateInsertValue(" + op1 + ", " + op2 + ", " + argStr + ")";
  } break;
  case Instruction::LandingPad: {
    auto lpI = dyn_cast<LandingPadInst>(I);
    std::string lpVar = getVal(I);
    call = "CreateLandingPad(" + getType(I->getType()) + ", " +
           std::to_string(lpI->getNumClauses()) + ")";
    OS << "auto " << lpVar << " = " << builder << "." << call << ";\n";
    for (unsigned i = 0; i < lpI->getNumClauses(); ++i) {
      OS << lpVar << "->addClause(" << getVal(lpI->getClause(i)) << ");\n";
    }
    return;
  } break;
  case Instruction::Freeze: {
    call = "CreateFreeze(" + op1 + ")";
  } break;
  default:
    OS << "// Unknown instruction: " << *I << "\n";
    return;
  }

  // Check if call returns a value
  if (!I->getType()->isVoidTy()) {
    OS << "auto " << getVal(I) << " = ";
  }
  OS << builder << "." << call << ";\n";
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  cl::HideUnrelatedOptions(Ir2BCat);
  cl::ParseCommandLineOptions(argc, argv);

  // Parse input file
  LLVMContext Context;
  SMDiagnostic Err;
  auto SetDataLayout = [](StringRef, StringRef) -> std::optional<std::string> {
    if (ClDataLayout.empty())
      return std::nullopt;
    return ClDataLayout;
  };
  ParsedModuleAndIndex ModuleAndIndex;
  if (DisableVerify) {
    ModuleAndIndex = parseAssemblyFileWithIndexNoUpgradeDebugInfo(
        InputFilename, Err, Context, nullptr, SetDataLayout);
  } else {
    ModuleAndIndex = parseAssemblyFileWithIndex(InputFilename, Err, Context,
                                                nullptr, SetDataLayout);
  }
  std::unique_ptr<Module> M = std::move(ModuleAndIndex.Mod);
  if (!M.get()) {
    Err.print(argv[0], errs());
    return 1;
  }

  // Output generation
  IR2Builder ir2b;
  if (!OutputFilename.empty()) {
    std::error_code EC;
    std::unique_ptr<ToolOutputFile> Out(
        new ToolOutputFile(OutputFilename, EC, sys::fs::OF_None));
    if (EC) {
      errs() << EC.message() << '\n';
      exit(1);
    }
    ir2b.convert(*(M.get()), Out->os());
    Out->keep();
  } else {
    ir2b.convert(*(M.get()), outs());
  }

  return 0;
}
