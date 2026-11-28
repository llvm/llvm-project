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
#include "llvm/Support/FormatVariadic.h"
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

static cl::OptionCategory Ir2BCat("ir2builder Options");

/// LLVM IR to convert.
static cl::opt<std::string>
    InputFilename(cl::Positional, cl::desc("<input .ll file>"), cl::init("-"));

/// Output file for the generated C++ code.
/// If not set then stdout will be used.
static cl::opt<std::string> OutputFilename("o",
                                           cl::desc("Override output filename"),
                                           cl::value_desc("filename"),
                                           cl::cat(Ir2BCat));

/// Set this when you don't use `using namespace llvm` (don't forget :: at the
/// end).
static cl::opt<std::string> ScopePrefix(
    "scope-prefix",
    cl::desc(
        "All generated calls to LLVM API will be prefixed with this scope. The "
        "scope has to end with '::' (e.g.: '-scope-prefix=llvm::')"),
    cl::cat(Ir2BCat), cl::init(""));

/// Set this to your variable name for IRBuilder instance.
static cl::opt<std::string> BuilderName(
    "builder-name",
    cl::desc("IRBuilder variable name that will be used in generated code"),
    cl::cat(Ir2BCat), cl::init("Builder"));

/// Set this to your LLVMContext variable name.
static cl::opt<std::string> ContextName(
    "context-name",
    cl::desc("Context variable name that will be used in generated code"),
    cl::cat(Ir2BCat), cl::init("Ctx"));

/// Set this to your llvm::Module * name.
static cl::opt<std::string> ModuleName(
    "module-name",
    cl::desc("Module variable name that will be used in generated code"),
    cl::cat(Ir2BCat), cl::init("Mod"));

/// Set this if you want custom data layout.
static cl::opt<std::string> ClDataLayout("data-layout",
                                         cl::desc("data layout string to use"),
                                         cl::value_desc("layout-string"),
                                         cl::init(""), cl::cat(Ir2BCat));

/// This will generate fully compilable C++ program with main, which will
/// use generated calls to create the original LLVM IR.
/// Main purpose of this is to use it for testing and verification that
/// generated program is correct and compilable.
/// For example you can generate the code, run this program and then use
/// llvm-diff to see if it matches the original:
///    ```
///    ./ir2builder code.ll --runnable > main.cpp &&
///    g++ main.cpp `./llvm-conf --cxxflags --ldflags --system-libs --libs core`
/// -I /llvm/include/ -o main &&
///    ./main > generated.ll &&
///    ./llvm-diff code.ll generated.ll
///    ```
static cl::opt<bool> GenerateRunnable(
    "runnable", cl::desc("Generates whole cpp compilable program with main"),
    cl::init(false), cl::cat(Ir2BCat));

/// Disables verification of loaded llvm IR.
/// Keep in mind that this will most likely result in C++ error as it probably
/// won't be possible to create Builder calls for this.
static cl::opt<bool>
    DisableVerify("disable-verify", cl::Hidden,
                  cl::desc("Do not run verifier on input LLVM (dangerous!)"),
                  cl::cat(Ir2BCat));

/// Sets the order of traversal for the LLVM IR.
/// When enabled the traversal will be in reverse post order, which can handle
/// when values are defined after (text-wise) their use.
/// On the other hand using just linear traversal will also include parts that
/// are outside of the graph (dead blocks).
static cl::opt<bool>
    UseRPO("use-rpo",
           cl::desc("Traverses IR in reverse post order. This can help with "
                    "\"was not declared\" errors"),
           cl::init(true), cl::cat(Ir2BCat));

namespace {
/// \brief Transpiler from LLVM IR into IRBuilder API calls.
/// The main purpose for this class is to hold variable counter and variable
/// names for needed resources, such as LLVMContext.
class IR2Builder {
private:
  unsigned long VarI = 0;
  std::string LLVMPrefix;
  std::string Builder;
  std::string Ctx;
  std::string ModName;

  std::vector<std::string> PhiIncomings;

  inline bool hasName(const Value *Op) { return !isa<Constant, InlineAsm>(Op); }

  void outputAttr(Attribute Att, raw_ostream &OS);

  std::string getNextVar();

  std::string asStr(GlobalValue::LinkageTypes Linkage);
  std::string asStr(GlobalValue::ThreadLocalMode TLM);
  std::string asStr(CmpInst::Predicate P);
  std::string asStr(AtomicRMWInst::BinOp Op);
  std::string asStr(AtomicOrdering AO);
  std::string asStr(SyncScope::ID Sys);
  std::string asStr(CallingConv::ID CC);
  std::string asStr(ConstantRange &CR);

  std::string asStr(const Value *Op);
  std::string asStr(const Constant *Op);
  std::string asStr(const Type *T);
  std::string asStr(const InlineAsm *Op);
  std::string asStr(const Metadata *Op);

public:
  IR2Builder()
      : LLVMPrefix(ScopePrefix), Builder(BuilderName), Ctx(ContextName),
        ModName(ModuleName) {}

  /// Calls convert for all the functions in passed in module.
  /// \param M Module to call convert over.
  /// \param OS Stream to which output the builder calls.
  void convert(Module &M, raw_ostream &OS);

  /// Converts a function into IRBuilder API calls.
  /// \param F Function to convert.
  /// \param OS Stream to which output the builder calls.
  void convert(Function &F, raw_ostream &OS);

  /// Converts an instruction into IRBuilder API calls.
  /// \param I Instruction to convert.
  /// \param OS Stream to which output the builder calls.
  /// \note Unsupported instructions or their operands should result
  ///       in a TODO comment.
  void convert(const Instruction *I, raw_ostream &OS);
};
} // namespace

std::string IR2Builder::getNextVar() { return "v0" + std::to_string(VarI++); }

static std::string toStr(bool B) { return B ? "true" : "false"; }

static std::string escape(std::string S) {
  std::string Tmp;
  raw_string_ostream OS(Tmp);
  printEscapedString(S, OS);
  return Tmp;
}

static std::string sanitize(std::string &S) {
  std::stringstream SS;
  for (auto C : S) {
    if (!std::isalnum(C))
      SS << "_" << static_cast<unsigned>(C) << "_";
    else
      SS << C;
  }
  return SS.str();
}

std::string IR2Builder::asStr(GlobalValue::LinkageTypes Linkage) {
  std::string Link = LLVMPrefix + "GlobalValue::";

  switch (Linkage) {
  case GlobalValue::WeakODRLinkage:
    return Link + "WeakODRLinkage";
  case GlobalValue::LinkOnceODRLinkage:
    return Link + "LinkOnceODRLinkage";
  case GlobalValue::AvailableExternallyLinkage:
    return Link + "AvailableExternallyLinkage";
  case GlobalValue::WeakAnyLinkage:
    return Link + "WeakAnyLinkage";
  case GlobalValue::LinkOnceAnyLinkage:
    return Link + "LinkOnceAnyLinkage";
  case GlobalValue::CommonLinkage:
    return Link + "CommonLinkage";
  case GlobalValue::ExternalWeakLinkage:
    return Link + "ExternalWeakLinkage";
  case GlobalValue::ExternalLinkage:
    return Link + "ExternalLinkage";
  case GlobalValue::AppendingLinkage:
    return Link + "AppendingLinkage";
  case GlobalValue::InternalLinkage:
    return Link + "InternalLinkage";
  case GlobalValue::PrivateLinkage:
    return Link + "PrivateLinkage";
  default:
    return "/* Unknown LinkageTypes (using value) */" + std::to_string(Linkage);
  }
}

std::string IR2Builder::asStr(AtomicRMWInst::BinOp Op) {
  switch (Op) {
  case AtomicRMWInst::BinOp::Xchg:
    return LLVMPrefix + "AtomicRMWInst::BinOp::Xchg";
  case AtomicRMWInst::BinOp::Add:
    return LLVMPrefix + "AtomicRMWInst::BinOp::Add";
  case AtomicRMWInst::BinOp::Sub:
    return LLVMPrefix + "AtomicRMWInst::BinOp::Sub";
  case AtomicRMWInst::BinOp::And:
    return LLVMPrefix + "AtomicRMWInst::BinOp::And";
  case AtomicRMWInst::BinOp::Nand:
    return LLVMPrefix + "AtomicRMWInst::BinOp::Nand";
  case AtomicRMWInst::BinOp::Or:
    return LLVMPrefix + "AtomicRMWInst::BinOp::Or";
  case AtomicRMWInst::BinOp::Xor:
    return LLVMPrefix + "AtomicRMWInst::BinOp::Xor";
  case AtomicRMWInst::BinOp::Max:
    return LLVMPrefix + "AtomicRMWInst::BinOp::Max";
  case AtomicRMWInst::BinOp::Min:
    return LLVMPrefix + "AtomicRMWInst::BinOp::Min";
  case AtomicRMWInst::BinOp::UMax:
    return LLVMPrefix + "AtomicRMWInst::BinOp::UMax";
  case AtomicRMWInst::BinOp::UMin:
    return LLVMPrefix + "AtomicRMWInst::BinOp::UMin";
  case AtomicRMWInst::BinOp::FAdd:
    return LLVMPrefix + "AtomicRMWInst::BinOp::FAdd";
  case AtomicRMWInst::BinOp::FSub:
    return LLVMPrefix + "AtomicRMWInst::BinOp::FSub";
  case AtomicRMWInst::BinOp::FMax:
    return LLVMPrefix + "AtomicRMWInst::BinOp::FMax";
  case AtomicRMWInst::BinOp::FMin:
    return LLVMPrefix + "AtomicRMWInst::BinOp::FMin";
  case AtomicRMWInst::BinOp::UIncWrap:
    return LLVMPrefix + "AtomicRMWInst::BinOp::UIncWrap";
  case AtomicRMWInst::BinOp::UDecWrap:
    return LLVMPrefix + "AtomicRMWInst::BinOp::UDecWrap";
  default:
    return "/* TODO: Unknown AtomicRMW operator (using value) */ " +
           std::to_string(Op);
  }
}

std::string IR2Builder::asStr(AtomicOrdering AO) {
  switch (AO) {
  case AtomicOrdering::NotAtomic:
    return LLVMPrefix + "AtomicOrdering::NotAtomic";
  case AtomicOrdering::Unordered:
    return LLVMPrefix + "AtomicOrdering::Unordered";
  case AtomicOrdering::Monotonic:
    return LLVMPrefix + "AtomicOrdering::Monotonic";
  case AtomicOrdering::Acquire:
    return LLVMPrefix + "AtomicOrdering::Acquire";
  case AtomicOrdering::Release:
    return LLVMPrefix + "AtomicOrdering::Release";
  case AtomicOrdering::AcquireRelease:
    return LLVMPrefix + "AtomicOrdering::AcquireRelease";
  case AtomicOrdering::SequentiallyConsistent:
    return LLVMPrefix + "AtomicOrdering::SequentiallyConsistent";
  default:
    return "/* TODO: Unknown atomic ordering */";
  }
}

std::string IR2Builder::asStr(CmpInst::Predicate P) {
  switch (P) {
  case CmpInst::Predicate::FCMP_FALSE:
    return LLVMPrefix + "CmpInst::Predicate::FCMP_FALSE";
  case CmpInst::Predicate::FCMP_OEQ:
    return LLVMPrefix + "CmpInst::Predicate::FCMP_OEQ";
  case CmpInst::Predicate::FCMP_OGT:
    return LLVMPrefix + "CmpInst::Predicate::FCMP_OGT";
  case CmpInst::Predicate::FCMP_OGE:
    return LLVMPrefix + "CmpInst::Predicate::FCMP_OGE";
  case CmpInst::Predicate::FCMP_OLT:
    return LLVMPrefix + "CmpInst::Predicate::FCMP_OLT";
  case CmpInst::Predicate::FCMP_OLE:
    return LLVMPrefix + "CmpInst::Predicate::FCMP_OLE";
  case CmpInst::Predicate::FCMP_ONE:
    return LLVMPrefix + "CmpInst::Predicate::FCMP_ONE";
  case CmpInst::Predicate::FCMP_ORD:
    return LLVMPrefix + "CmpInst::Predicate::FCMP_ORD";
  case CmpInst::Predicate::FCMP_UNO:
    return LLVMPrefix + "CmpInst::Predicate::FCMP_UNO";
  case CmpInst::Predicate::FCMP_UEQ:
    return LLVMPrefix + "CmpInst::Predicate::FCMP_UEQ";
  case CmpInst::Predicate::FCMP_UGT:
    return LLVMPrefix + "CmpInst::Predicate::FCMP_UGT";
  case CmpInst::Predicate::FCMP_UGE:
    return LLVMPrefix + "CmpInst::Predicate::FCMP_UGE";
  case CmpInst::Predicate::FCMP_ULT:
    return LLVMPrefix + "CmpInst::Predicate::FCMP_ULT";
  case CmpInst::Predicate::FCMP_ULE:
    return LLVMPrefix + "CmpInst::Predicate::FCMP_ULE";
  case CmpInst::Predicate::FCMP_UNE:
    return LLVMPrefix + "CmpInst::Predicate::FCMP_UNE";
  case CmpInst::Predicate::FCMP_TRUE:
    return LLVMPrefix + "CmpInst::Predicate::FCMP_TRUE";
  case CmpInst::Predicate::ICMP_EQ:
    return LLVMPrefix + "CmpInst::Predicate::ICMP_EQ";
  case CmpInst::Predicate::ICMP_NE:
    return LLVMPrefix + "CmpInst::Predicate::ICMP_NE";
  case CmpInst::Predicate::ICMP_UGT:
    return LLVMPrefix + "CmpInst::Predicate::ICMP_UGT";
  case CmpInst::Predicate::ICMP_UGE:
    return LLVMPrefix + "CmpInst::Predicate::ICMP_UGE";
  case CmpInst::Predicate::ICMP_ULT:
    return LLVMPrefix + "CmpInst::Predicate::ICMP_ULT";
  case CmpInst::Predicate::ICMP_ULE:
    return LLVMPrefix + "CmpInst::Predicate::ICMP_ULE";
  case CmpInst::Predicate::ICMP_SGT:
    return LLVMPrefix + "CmpInst::Predicate::ICMP_SGT";
  case CmpInst::Predicate::ICMP_SGE:
    return LLVMPrefix + "CmpInst::Predicate::ICMP_SGE";
  case CmpInst::Predicate::ICMP_SLT:
    return LLVMPrefix + "CmpInst::Predicate::ICMP_SLT";
  case CmpInst::Predicate::ICMP_SLE:
    return LLVMPrefix + "CmpInst::Predicate::ICMP_SLE";
  default:
    return "/* TODO: Unknown CMP predicate (using value) */ " +
           std::to_string(P);
  }
}

std::string IR2Builder::asStr(SyncScope::ID Sys) {
  if (Sys == SyncScope::System)
    return LLVMPrefix + "SyncScope::System";
  if (Sys == SyncScope::SingleThread)
    return LLVMPrefix + "SyncScope::SingleThread";

  return "/* TODO: Unknown SyncScope ID (using value) */ " +
         std::to_string(Sys);
}

std::string IR2Builder::asStr(GlobalValue::ThreadLocalMode TLM) {
  switch (TLM) {
  case GlobalValue::ThreadLocalMode::NotThreadLocal:
    return LLVMPrefix + "GlobalValue::ThreadLocalMode::NotThreadLocal";
  case GlobalValue::ThreadLocalMode::GeneralDynamicTLSModel:
    return LLVMPrefix + "GlobalValue::ThreadLocalMode::GeneralDynamicTLSModel";
  case GlobalValue::ThreadLocalMode::LocalDynamicTLSModel:
    return LLVMPrefix + "GlobalValue::ThreadLocalMode::LocalDynamicTLSModel";
  case GlobalValue::ThreadLocalMode::InitialExecTLSModel:
    return LLVMPrefix + "GlobalValue::ThreadLocalMode::InitialExecTLSModel";
  case GlobalValue::ThreadLocalMode::LocalExecTLSModel:
    return LLVMPrefix + "GlobalValue::ThreadLocalMode::LocalExecTLSModel";
  default:
    return "/* TODO: Unknown ThreadLocalMode (using value) */ " +
           std::to_string(TLM);
  }
}

std::string IR2Builder::asStr(CallingConv::ID CC) {
  switch (CC) {
  case CallingConv::C:
    return LLVMPrefix + "CallingConv::C";
  case CallingConv::Fast:
    return LLVMPrefix + "CallingConv::Fast";
  case CallingConv::Cold:
    return LLVMPrefix + "CallingConv::Cold";
  case CallingConv::GHC:
    return LLVMPrefix + "CallingConv::GHC";
  case CallingConv::HiPE:
    return LLVMPrefix + "CallingConv::HiPE";
  case CallingConv::AnyReg:
    return LLVMPrefix + "CallingConv::AnyReg";
  case CallingConv::PreserveMost:
    return LLVMPrefix + "CallingConv::PreserveMost";
  case CallingConv::PreserveAll:
    return LLVMPrefix + "CallingConv::PreserveAll";
  case CallingConv::Swift:
    return LLVMPrefix + "CallingConv::Swift";
  case CallingConv::CXX_FAST_TLS:
    return LLVMPrefix + "CallingConv::CXX_FAST_TLS";
  case CallingConv::Tail:
    return LLVMPrefix + "CallingConv::Tail";
  case CallingConv::CFGuard_Check:
    return LLVMPrefix + "CallingConv::CFGuard_Check";
  case CallingConv::SwiftTail:
    return LLVMPrefix + "CallingConv::SwiftTail";
  case CallingConv::PreserveNone:
    return LLVMPrefix + "CallingConv::PreserveNone";
  case CallingConv::FirstTargetCC:
    return LLVMPrefix + "CallingConv::FirstTargetCC";
  // CallingConv::X86_StdCall is the same as FirstTargetCC.
  case CallingConv::X86_FastCall:
    return LLVMPrefix + "CallingConv::X86_FastCall";
  case CallingConv::ARM_APCS:
    return LLVMPrefix + "CallingConv::ARM_APCS";
  case CallingConv::ARM_AAPCS:
    return LLVMPrefix + "CallingConv::ARM_AAPCS";
  case CallingConv::ARM_AAPCS_VFP:
    return LLVMPrefix + "CallingConv::ARM_AAPCS_VFP";
  case CallingConv::MSP430_INTR:
    return LLVMPrefix + "CallingConv::MSP430_INTR";
  case CallingConv::X86_ThisCall:
    return LLVMPrefix + "CallingConv::X86_ThisCall";
  case CallingConv::PTX_Kernel:
    return LLVMPrefix + "CallingConv::PTX_Kernel";
  case CallingConv::PTX_Device:
    return LLVMPrefix + "CallingConv::PTX_Device";
  case CallingConv::SPIR_FUNC:
    return LLVMPrefix + "CallingConv::SPIR_FUNC";
  case CallingConv::SPIR_KERNEL:
    return LLVMPrefix + "CallingConv::SPIR_KERNEL";
  case CallingConv::Intel_OCL_BI:
    return LLVMPrefix + "CallingConv::Intel_OCL_BI";
  case CallingConv::X86_64_SysV:
    return LLVMPrefix + "CallingConv::X86_64_SysV";
  case CallingConv::Win64:
    return LLVMPrefix + "CallingConv::Win64";
  case CallingConv::X86_VectorCall:
    return LLVMPrefix + "CallingConv::X86_VectorCall";
  case CallingConv::DUMMY_HHVM:
    return LLVMPrefix + "CallingConv::DUMMY_HHVM";
  case CallingConv::DUMMY_HHVM_C:
    return LLVMPrefix + "CallingConv::DUMMY_HHVM_C";
  case CallingConv::X86_INTR:
    return LLVMPrefix + "CallingConv::X86_INTR";
  case CallingConv::AVR_INTR:
    return LLVMPrefix + "CallingConv::AVR_INTR";
  case CallingConv::AVR_SIGNAL:
    return LLVMPrefix + "CallingConv::AVR_SIGNAL";
  case CallingConv::AVR_BUILTIN:
    return LLVMPrefix + "CallingConv::AVR_BUILTIN";
  case CallingConv::AMDGPU_VS:
    return LLVMPrefix + "CallingConv::AMDGPU_VS";
  case CallingConv::AMDGPU_GS:
    return LLVMPrefix + "CallingConv::AMDGPU_GS";
  case CallingConv::AMDGPU_PS:
    return LLVMPrefix + "CallingConv::AMDGPU_PS";
  case CallingConv::AMDGPU_CS:
    return LLVMPrefix + "CallingConv::AMDGPU_CS";
  case CallingConv::AMDGPU_KERNEL:
    return LLVMPrefix + "CallingConv::AMDGPU_KERNEL";
  case CallingConv::X86_RegCall:
    return LLVMPrefix + "CallingConv::X86_RegCall";
  case CallingConv::AMDGPU_HS:
    return LLVMPrefix + "CallingConv::AMDGPU_HS";
  case CallingConv::MSP430_BUILTIN:
    return LLVMPrefix + "CallingConv::MSP430_BUILTIN";
  case CallingConv::AMDGPU_LS:
    return LLVMPrefix + "CallingConv::AMDGPU_LS";
  case CallingConv::AMDGPU_ES:
    return LLVMPrefix + "CallingConv::AMDGPU_ES";
  case CallingConv::AArch64_VectorCall:
    return LLVMPrefix + "CallingConv::AArch64_VectorCall";
  case CallingConv::AArch64_SVE_VectorCall:
    return LLVMPrefix + "CallingConv::AArch64_SVE_VectorCall";
  case CallingConv::WASM_EmscriptenInvoke:
    return LLVMPrefix + "CallingConv::WASM_EmscriptenInvoke";
  case CallingConv::AMDGPU_Gfx:
    return LLVMPrefix + "CallingConv::AMDGPU_Gfx";
  case CallingConv::M68k_INTR:
    return LLVMPrefix + "CallingConv::M68k_INTR";
  case CallingConv::AArch64_SME_ABI_Support_Routines_PreserveMost_From_X0:
    return LLVMPrefix +
           "CallingConv::AArch64_SME_ABI_Support_Routines_PreserveMost_From_X0";
  case CallingConv::AArch64_SME_ABI_Support_Routines_PreserveMost_From_X2:
    return LLVMPrefix +
           "CallingConv::AArch64_SME_ABI_Support_Routines_PreserveMost_From_X2";
  case CallingConv::AMDGPU_CS_Chain:
    return LLVMPrefix + "CallingConv::AMDGPU_CS_Chain";
  case CallingConv::AMDGPU_CS_ChainPreserve:
    return LLVMPrefix + "CallingConv::AMDGPU_CS_ChainPreserve";
  case CallingConv::M68k_RTD:
    return LLVMPrefix + "CallingConv::M68k_RTD";
  case CallingConv::GRAAL:
    return LLVMPrefix + "CallingConv::GRAAL";
  case CallingConv::ARM64EC_Thunk_X64:
    return LLVMPrefix + "CallingConv::ARM64EC_Thunk_X64";
  case CallingConv::ARM64EC_Thunk_Native:
    return LLVMPrefix + "CallingConv::ARM64EC_Thunk_Native";
  default:
    return "/* Custom CC */" + std::to_string(CC);
  }
}

std::string IR2Builder::asStr(const Type *T) {
  std::string TCall;
  if (auto *IT = dyn_cast<IntegerType>(T)) {
    switch (IT->getBitWidth()) {
    case 1:
    case 8:
    case 16:
    case 32:
    case 64:
      TCall = formatv("getInt{0}Ty()", IT->getBitWidth());
      break;
    default:
      TCall = formatv("getIntNTy({0})", IT->getBitWidth());
      break;
    }
  } else if (T->isVoidTy())
    TCall = "getVoidTy()";
  else if (T->isFloatTy())
    TCall = "getFloatTy()";
  else if (T->isDoubleTy())
    TCall = "getDoubleTy()";
  else if (T->isHalfTy())
    TCall = "getHalfTy()";
  else if (T->isBFloatTy())
    TCall = "getBFloatTy()";
  else if (auto ST = dyn_cast<StructType>(T)) {
    std::string Elements = LLVMPrefix + "ArrayRef<Type *>(";
    if (ST->getNumElements() > 1)
      Elements += "{";
    SmallVector<std::string> Tmp;
    Elements +=
        formatv("{0}", (llvm::transform(ST->elements(), back_inserter(Tmp),
                                        [&](Type *V) { return asStr(V); }),
                        make_range(Tmp.begin(), Tmp.end())));
    if (ST->getNumElements() > 1)
      Elements += "}";
    Elements += ")";
    return formatv("{0}StructType::create({1}, {2})", LLVMPrefix, Ctx,
                   Elements);
  } else if (auto AT = dyn_cast<ArrayType>(T)) {
    return formatv("{0}ArrayType::get({1}, {2})", LLVMPrefix,
                   asStr(AT->getElementType()), AT->getArrayNumElements());
  } else if (T->isBFloatTy())
    TCall = "getBFloatTy()";
  else if (auto VT = dyn_cast<VectorType>(T)) {
    return formatv("{0}VectorType::get({1}, {0}ElementCount::get({2}, {3}))",
                   LLVMPrefix, asStr(VT->getElementType()),
                   VT->getElementCount().getKnownMinValue(),
                   toStr(VT->getElementCount().isScalable()));
  } else if (auto FT = dyn_cast<FunctionType>(T)) {
    TCall = formatv("{0}FunctionType::get({1}, {{", LLVMPrefix,
                    asStr(FT->getReturnType()));
    SmallVector<std::string> Tmp;
    TCall += formatv("{0}", (llvm::transform(FT->params(), back_inserter(Tmp),
                                             [&](Type *V) { return asStr(V); }),
                             make_range(Tmp.begin(), Tmp.end())));
    TCall += "}, " + toStr(FT->isVarArg()) + ")";
    return TCall;
  } else if (T->isPointerTy())
    TCall = "getPtrTy()";
  else if (T->isX86_FP80Ty())
    return formatv("{0}Type::getX86_FP80Ty({1})", LLVMPrefix, Ctx);
  else if (T->isFP128Ty())
    return formatv("{0}Type::getFP128Ty({1})", LLVMPrefix, Ctx);
  else if (T->isPPC_FP128Ty())
    return formatv("{0}Type::getPPC_FP128Ty({1})", LLVMPrefix, Ctx);
  else if (T->isX86_AMXTy())
    return formatv("{0}Type::getX86_AMXTy({1})", LLVMPrefix, Ctx);
  else if (T->isLabelTy())
    return formatv("{0}Type::getLabelTy({1})", LLVMPrefix, Ctx);
  else if (T->isMetadataTy())
    return formatv("{0}Type::getMetadataTy({1})", LLVMPrefix, Ctx);
  else if (T->isTokenTy())
    return formatv("{0}Type::getTokenTy({1})", LLVMPrefix, Ctx);
  else
    return "/* TODO: Unknown type */";

  return Builder + "." + TCall;
}

std::string IR2Builder::asStr(const Constant *C) {
  if (auto *CI = dyn_cast<ConstantInt>(C)) {
    // TODO: Sign has to be determined.
    auto CVal = CI->getValue();
    return formatv("{0}ConstantInt::get({1}, {2})", LLVMPrefix,
                   asStr(C->getType()), CVal.getSExtValue());
  }
  if (auto *CF = dyn_cast<ConstantFP>(C)) {
    auto CVal = CF->getValue();
    double DVal = CVal.convertToDouble();
    std::string Val = std::to_string(DVal);
    if (std::isnan(DVal) || std::isinf(DVal))
      Val = "\"" + Val + "\"";
    // TODO: Handle double to string conversion to include all digits.
    return formatv("{0}ConstantFP::get({1}, {2})", LLVMPrefix,
                   asStr(C->getType()), Val);
  }
  if (auto *AT = dyn_cast<ConstantAggregate>(C)) {
    SmallVector<std::string> Tmp;
    std::string Values =
        formatv("{0}", (llvm::transform(AT->operands(), back_inserter(Tmp),
                                        [&](Value *V) { return asStr(V); }),
                        make_range(Tmp.begin(), Tmp.end())));
    std::string ClassName;
    if (isa<ConstantArray>(C)) {
      ClassName = "ConstantArray";
      Values = formatv(
          "{0}ArrayRef<{0}Constant *>({1}{2}{3})", LLVMPrefix,
          (AT->getNumOperands() > 1 ? std::string("{") : std::string("")),
          Values,
          (AT->getNumOperands() > 1 ? std::string("}") : std::string("")));
    } else if (isa<ConstantStruct>(C))
      ClassName = "ConstantStruct";
    else if (isa<ConstantVector>(C)) {
      Values = "{" + Values + "}";
      // ConstantVector does not take type as 1st arg.
      return formatv("{0}ConstantVector::get({1})", LLVMPrefix, Values);
    } else
      return "/* TODO: Unknown aggregate constant */";

    assert(!ClassName.empty() && "Class name not set");
    return formatv("{0}{1}::get({2}, {3})", LLVMPrefix, ClassName,
                   asStr(C->getType()), Values);
  }
  if (auto CDS = dyn_cast<ConstantDataSequential>(C)) {
    std::string Values;
    std::string ClassName;
    std::string ElemTy = "/* TODO */";
    if (isa<ConstantDataArray>(C))
      ClassName = "ConstantDataArray";
    else if (isa<ConstantDataVector>(C))
      ClassName = "ConstantDataVector";
    else
      return "/* TODO: Unknown data sequential constant */";
    if (CDS->isString()) {
      SmallVector<std::string> Tmp;
      Values = formatv(
          "{0}",
          (llvm::transform(
               CDS->getAsString().str(), back_inserter(Tmp),
               [&](char V) { return std::to_string(static_cast<uint8_t>(V)); }),
           make_range(Tmp.begin(), Tmp.end())));
      assert(!ClassName.empty() && "Class name not set");
      return formatv("{0}{1}::get({2}, {0}ArrayRef<uint8_t>({3}{4}{5}))",
                     LLVMPrefix, ClassName, Ctx, "{", Values, "}");
    }
    if (CDS->isCString()) {
      SmallVector<std::string> Tmp;
      Values = formatv(
          "{0}",
          (llvm::transform(
               CDS->getAsCString().str(), back_inserter(Tmp),
               [&](char V) { return std::to_string(static_cast<uint8_t>(V)); }),
           make_range(Tmp.begin(), Tmp.end())));
      assert(!ClassName.empty() && "Class name not set");
      return formatv("{0}{1}::get({2}, {0}ArrayRef<uint8_t>({3}{4}{5}))",
                     LLVMPrefix, ClassName, Ctx, "{", Values, "}");
    } else {
      Type *ElemT = CDS->getElementType();
      if (ElemT->isIntegerTy()) {
        // There can be only 8, 16, 32 or 64 ints in ConstantDataVector.
        ElemTy = "uint" + std::to_string(ElemT->getIntegerBitWidth()) + "_t";
      } else if (ElemT->isDoubleTy()) {
        ElemTy = "double";
      } else if (ElemT->isFloatTy()) {
        ElemTy = "float";
      }
      Values = formatv("{0}ArrayRef<{1}>(", LLVMPrefix, ElemTy);
      if (CDS->getNumElements() > 1)
        Values += "{";
      bool First = true;
      for (unsigned I = 0; I < CDS->getNumElements(); ++I) {
        if (!First)
          Values += ", ";
        if (ElemT->isIntegerTy())
          Values += std::to_string(CDS->getElementAsInteger(I));
        else if (ElemT->isDoubleTy())
          Values += std::to_string(CDS->getElementAsDouble(I));
        else if (ElemT->isFloatTy())
          Values += std::to_string(CDS->getElementAsFloat(I));
        else
          return "/* Unknown type in data sequential constant */";
        First = false;
      }
      if (CDS->getNumElements() > 1)
        Values += "}";
      Values += ")";
    }

    return formatv("{0}{1}::get({2}, {3})", LLVMPrefix, ClassName, Ctx, Values);
  }
  if (isa<ConstantAggregateZero>(C))
    return formatv("{0}ConstantAggregateZero::get({1})", LLVMPrefix,
                   asStr(C->getType()));
  if (isa<PoisonValue>(C))
    return formatv("{0}PoisonValue::get({1})", LLVMPrefix, asStr(C->getType()));
  if (isa<UndefValue>(C))
    return formatv("{0}UndefValue::get({1})", LLVMPrefix, asStr(C->getType()));
  if (auto *BA = dyn_cast<BlockAddress>(C))
    return formatv("{0}BlockAddress::get({1}, {2})", LLVMPrefix,
                   asStr(BA->getFunction()), asStr(BA->getBasicBlock()));
  if (isa<ConstantPointerNull>(C))
    return formatv("{0}ConstantPointerNull::get({1})", LLVMPrefix,
                   asStr(C->getType()));
  if (auto *CTN = dyn_cast<ConstantTargetNone>(C)) {
    auto CTNType = CTN->getType();
    SmallVector<std::string> Tmp;
    auto TypeStr =
        "{" +
        formatv("{0}",
                (llvm::transform(CTNType->type_params(), back_inserter(Tmp),
                                 [&](Type *V) { return asStr(V); }),
                 make_range(Tmp.begin(), Tmp.end()))) +
        "}";

    SmallVector<std::string> Tmp2;
    auto IntsStr =
        "{" +
        formatv("{0}",
                (llvm::transform(CTNType->int_params(), back_inserter(Tmp2),
                                 [&](unsigned V) { return std::to_string(V); }),
                 make_range(Tmp2.begin(), Tmp2.end()))) +
        "}";

    return formatv("{0}ConstantTargetNone::get({0}TargetExtType::get({1}, "
                   "\"{2}\", {3}, {4}))",
                   LLVMPrefix, Ctx, escape(CTNType->getName().str()), TypeStr,
                   IntsStr);
  }
  if (isa<ConstantTokenNone>(C)) {
    return formatv("{0}ConstantTokenNone::get({1})", LLVMPrefix, Ctx);
  }
  if (auto *CE = dyn_cast<ConstantExpr>(C)) {
    (void)CE;
    return "/* TODO: ConstantExpr creation */";
    // TODO: Dunno how to create this... Fails either on out of range or
    // even on incorrect opcode.
    // return LLVMPrefix + "ConstantExpr::get(" +
    // std::to_string(CE->getOpcode()) + ", " +
    //       asStr(CE->getOperand(0)) + ", " + asStr(CE->getOperand(1)) + ")";
  }
  if (auto *CE = dyn_cast<ConstantPtrAuth>(C)) {
    (void)CE;
    return "/* TODO: ConstantPtrAuth value creation */";
  }
  if (auto *CE = dyn_cast<DSOLocalEquivalent>(C)) {
    (void)CE;
    return "/* TODO: DSOLocalEquivalent value creation */";
  }
  if (auto *CE = dyn_cast<NoCFIValue>(C)) {
    (void)CE;
    return "/* TODO: NoCFIValue value creation */";
  }
  if (auto *CE = dyn_cast<GlobalValue>(C)) {
    // This should not really happen as asStr for Value should be always called.
    return asStr(cast<Value>(CE));
  }

  return "/* TODO: Constant creation */";
}

std::string IR2Builder::asStr(const InlineAsm *Op) {
  auto GetAsmDialect = [LLVMPrefix = LLVMPrefix](InlineAsm::AsmDialect D) {
    switch (D) {
    case InlineAsm::AsmDialect::AD_ATT:
      return LLVMPrefix + "InlineAsm::AsmDialect::AD_ATT";
    case InlineAsm::AsmDialect::AD_Intel:
      return LLVMPrefix + "InlineAsm::AsmDialect::AD_Intel";
    default:
      return "/* TODO: Unknown AsmDialect (using value) */" + std::to_string(D);
    }
  };

  return formatv("{0}InlineAsm::get({1}, \"{2}\", \"{3}\", {4}, {5}, {6}, {7})",
                 asStr(Op->getFunctionType()), escape(Op->getAsmString()),
                 escape(Op->getConstraintString()), toStr(Op->hasSideEffects()),
                 toStr(Op->isAlignStack()), GetAsmDialect(Op->getDialect()),
                 toStr(Op->canThrow()));
}

std::string IR2Builder::asStr(const Metadata *Op) {
  if (auto *MDN = dyn_cast<MDNode>(Op)) {
    std::string Args = "{";
    bool First = true;
    for (unsigned I = 0; I < MDN->getNumOperands(); ++I) {
      if (!First)
        Args += ", ";
      Args += asStr(MDN->getOperand(I));
      First = false;
    }
    Args += "}";
    return formatv("{0}MDNode::get({1}, {2})", LLVMPrefix, Ctx, Args);
  }
  if (auto *VAM = dyn_cast<ValueAsMetadata>(Op))
    return formatv("{0}ValueAsMetadata::get({1})", LLVMPrefix,
                   asStr(VAM->getValue()));
  if (auto *MDS = dyn_cast<MDString>(Op))
    return formatv("{0}MDString::get({1}, \"{2}\")", LLVMPrefix, Ctx,
                   escape(MDS->getString().str()));

  return "/* TODO: Metadata creation */";
}

// This is a DEBUG function in Value, so it's copied here for NDEBUG as well.
static std::string getNameOrAsOperand(const Value *V) {
  if (!V->getName().empty())
    return std::string(V->getName());

  std::string BBName;
  raw_string_ostream OS(BBName);
  V->printAsOperand(OS, false);
  return OS.str();
}

std::string IR2Builder::asStr(const Value *Op) {
  if (!Op)
    return "nullptr";
  if (isa<Constant>(Op) && !isa<GlobalValue>(Op))
    return asStr(cast<Constant>(Op));
  if (auto *InA = dyn_cast<InlineAsm>(Op))
    return asStr(InA);
  if (auto *Mtd = dyn_cast<MetadataAsValue>(Op)) {
    return formatv("{0}MetadataAsValue::get({1}, {2})", LLVMPrefix, Ctx,
                   asStr(Mtd->getMetadata()));
  }
  std::string OpName = getNameOrAsOperand(Op);
  if (OpName[0] == '%')
    OpName.erase(OpName.begin());
  std::string Pref = "v_";
  if (isa<GlobalValue>(Op))
    Pref = "g_";
  return Pref + sanitize(OpName);
}

std::string getBinArithOp(std::string Name, std::string Op1, std::string Op2,
                          const Instruction *I) {
  return formatv("Create{0}({1}, {2}, \"\", {3}, {4})", Name, Op1, Op2,
                 toStr(I->hasNoUnsignedWrap()), toStr(I->hasNoSignedWrap()));
}

std::string getFPBinArithOp(std::string Name, std::string Op1, std::string Op2,
                            const Instruction *I) {
  return formatv("Create{0}({1}, {2})", Name, Op1, Op2);
  // TODO: Handle FPMathTag.
}

std::string IR2Builder::asStr(ConstantRange &CR) {
  auto NumBitsStr = std::to_string(CR.getBitWidth());
  auto Lower = std::to_string(CR.getLower().getLimitedValue());
  auto Upper = std::to_string(CR.getUpper().getLimitedValue());
  return formatv(
      "{0}ConstantRange(APInt({1}, {2}, true), APInt({1}, {3}, true))",
      LLVMPrefix, NumBitsStr, Lower, Upper);
}

void IR2Builder::outputAttr(Attribute Att, raw_ostream &OS) {
  // TODO: Handle special cases detected using "has" methods
  // see Attribute::getAsString(bool InAttrGrp).
  if (Att.isStringAttribute()) {
    OS << LLVMPrefix << "Attribute::get(" << Ctx << ", \""
       << Att.getKindAsString() << "\"";
    auto Val = Att.getValueAsString();
    if (Val.empty()) {
      OS << ")";
    } else {
      OS << ", \"";
      printEscapedString(Val, OS);
      OS << "\")";
    }
  } else if (Att.isIntAttribute()) {
    OS << LLVMPrefix << "Attribute::get(" << Ctx << ", (" << LLVMPrefix
       << "Attribute::AttrKind)" << std::to_string(Att.getKindAsEnum())
       << ", static_cast<uint64_t>(" << Att.getValueAsInt() << "))";
  } else if (Att.isEnumAttribute()) {
    OS << LLVMPrefix << "Attribute::get(" << Ctx << ", \""
       << Att.getNameFromAttrKind(Att.getKindAsEnum()) << "\")";
  } else if (Att.isTypeAttribute()) {
    OS << LLVMPrefix << "Attribute::get(" << Ctx << ", (" << LLVMPrefix
       << "Attribute::AttrKind)" << std::to_string(Att.getKindAsEnum()) << ", "
       << asStr(Att.getValueAsType()) << ")";
  } else if (Att.isConstantRangeAttribute()) {
    auto CR = Att.getValueAsConstantRange();
    OS << LLVMPrefix << "Attribute::get(" << Ctx << ", (" << LLVMPrefix
       << "Attribute::AttrKind)" << std::to_string(Att.getKindAsEnum()) << ", "
       << asStr(CR) << ")";
  } else if (Att.isConstantRangeListAttribute()) {
    SmallVector<std::string> Tmp;
    auto Args =
        "{" +
        formatv("{0}",
                (llvm::transform(Att.getValueAsConstantRangeList(),
                                 back_inserter(Tmp),
                                 [&](ConstantRange V) { return asStr(V); }),
                 make_range(Tmp.begin(), Tmp.end()))) +
        "}";
    OS << LLVMPrefix << "Attribute::get(" << Ctx << ", (" << LLVMPrefix
       << "Attribute::AttrKind)" << std::to_string(Att.getKindAsEnum()) << ", "
       << LLVMPrefix << Args << ")";
  } else {
    OS << "/* TODO: Attribute creation */";
  }
}

void IR2Builder::convert(Module &M, raw_ostream &OS) {
  // Prologue.
  if (GenerateRunnable) {
    // Top comment.
    OS << "// This file was autogenerated using ir2builder tool\n";

    // Includes.
    // Currently we include all possibly needed files as this is done
    // before the conversion of functions.
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

    // Used objects and variables.
    OS << "    LLVMContext " << Ctx << ";\n"
       << "    Module *" << ModName
       << "= new Module(\"ir2builder generated module\", " << Ctx << ");\n"
       << "    IRBuilder<> " << Builder << "(" << Ctx << ");\n";

    // Globals generation.
    for (auto &G : M.globals()) {
      std::string InitVal = "nullptr";
      if (G.hasInitializer()) {
        // In case of BlockAddress initializer we need to declare the function
        // and basic block so it can be used.
        if (auto *BA = dyn_cast<BlockAddress>(G.getInitializer())) {
          OS << formatv("auto {0} = {1}->getOrInsertFunction(\"{2}\", {3});\n",
                        asStr(BA->getFunction()), ModName,
                        BA->getFunction()->getName(),
                        asStr(BA->getFunction()->getFunctionType()));
          // Extract basic block.
          // TODO: The function might need to be created in here
          // since the basic blocks don't exist yet.
          OS << "// TODO: Basic block extraction\n";
        }
        InitVal = asStr(G.getInitializer());
      }

      std::string GName = asStr(&G);
      OS << formatv("auto {0} = new {1}GlobalVariable(*{2}, {3}, {4}, {5}, "
                    "{6}, \"{7}\", nullptr, {8}, {9}, {10});\n",
                    GName, LLVMPrefix, ModName, asStr(G.getValueType()),
                    toStr(G.isConstant()), asStr(G.getLinkage()), InitVal,
                    escape(G.getName().str()), asStr(G.getThreadLocalMode()),
                    std::to_string(G.getAddressSpace()),
                    toStr(G.isExternallyInitialized()));
    }
  }

  // Function generation.
  for (Function &F : M) {
    if (!F.isDeclaration())
      convert(F, OS);
  }

  // Epilogue.
  if (GenerateRunnable) {
    // Print generated code.
    OS << "    " << ModName << "->print(outs(), nullptr);\n";
    // Cleanup.
    OS << "    delete " << ModName << ";\n"
       << "    return 0;\n}\n";
  }
}

void IR2Builder::convert(Function &F, raw_ostream &OS) {
  PhiIncomings.clear();
  // Function.
  OS << "{\n\n";
  auto FDecl = asStr(&F);
  OS << formatv(
      "{0}Function *{1} = {0}Function::Create({2}, {3}, \"{4}\", {5});\n\n",
      LLVMPrefix, FDecl, asStr(F.getFunctionType()), asStr(F.getLinkage()),
      F.getName(), ModName);

  // Set attributes.
  if (F.getCallingConv() != CallingConv::C) { // C is default.
    OS << FDecl << "->setCallingConv(" + asStr(F.getCallingConv()) << ");\n";
  }
  AttributeList AL = F.getAttributes();
  // TODO: Handle attributes with values.
  if (AL.hasFnAttrs()) {
    for (auto Att : AL.getFnAttrs()) {
      OS << FDecl << "->addFnAttr(";
      outputAttr(Att, OS);
      OS << ");\n";
    }
  }
  if (AL.hasRetAttrs()) {
    for (auto Att : AL.getRetAttrs()) {
      OS << FDecl << "->addRetAttr(";
      outputAttr(Att, OS);
      OS << ");\n";
    }
  }
  for (size_t I = 0; I < F.arg_size(); ++I) {
    if (AL.hasParamAttrs(I)) {
      for (auto Att : AL.getAttributes(I + 1)) {
        OS << FDecl << "->addParamAttr(" << I << ", ";
        outputAttr(Att, OS);
        OS << ");\n";
      }
    }
  }

  // Save arguments into variables for easy access.
  for (const auto &[I, Arg] : enumerate(F.args())) {
    OS << formatv("auto {0} = {1}->getArg({2});\n", asStr(&Arg), FDecl, I);
  }

  if (UseRPO) {
    ReversePostOrderTraversal<Function *> RPOT(&F);

    // Basic block declaration in order.
    for (BasicBlock *BB : RPOT) {
      OS << formatv(
          "{0}BasicBlock *{1} = {0}BasicBlock::Create({2}, \"{3}\", {4});\n",
          LLVMPrefix, asStr(BB), Ctx, BB->getName(), FDecl);
    }

    OS << "\n";

    for (auto *BB : RPOT) {
      OS << Builder << "."
         << "SetInsertPoint(" << asStr(BB) << ");\n";

      for (Instruction &Inst : *BB)
        convert(&Inst, OS);

      OS << "\n";
    }
  } else {
    for (BasicBlock &BB : F) {
      OS << formatv(
          "{0}BasicBlock *{1} = {0}BasicBlock::Create({2}, \"{3}\", {4});\n",
          LLVMPrefix, asStr(&BB), Ctx, BB.getName(), FDecl);
    }

    OS << "\n";

    for (auto &BB : F) {
      OS << Builder << "."
         << "SetInsertPoint(" << asStr(&BB) << ");\n";

      for (Instruction &Inst : BB)
        convert(&Inst, OS);

      OS << "\n";
    }
  }

  // Output incoming values assignment into phis, this is needed as they
  // might refer to a value not yet defined in the time of phi definition.
  for (auto Line : PhiIncomings) {
    OS << Line;
  }

  OS << "}\n";
}

void IR2Builder::convert(const Instruction *I, raw_ostream &OS) {
  std::string Call;

  std::string Op1 = "/* TODO */";
  std::string Op2 = "/* TODO */";
  std::string Op3 = "/* TODO */";

  if (I->getNumOperands() > 0)
    Op1 = asStr(I->getOperand(0));
  if (I->getNumOperands() > 1)
    Op2 = asStr(I->getOperand(1));
  if (I->getNumOperands() > 2)
    Op3 = asStr(I->getOperand(2));

  switch (I->getOpcode()) {
  case Instruction::Ret: {
    if (I->getNumOperands() == 0)
      Call = "CreateRetVoid()";
    else
      Call = "CreateRet(" + Op1 + ")";
  } break;
  case Instruction::Br: {
    const BranchInst *BI = cast<BranchInst>(I);
    if (BI->isUnconditional()) {
      Call = formatv("CreateBr({0})", Op1);
    } else {
      Call = formatv("CreateCondBr({0}, {1}, {2})", Op1, Op3, Op2);
    }
  } break;
  case Instruction::Switch: {
    auto *SwI = cast<SwitchInst>(I);
    Call = formatv("CreateSwitch({0}, {1}, {2})", Op1, Op2,
                   std::to_string(SwI->getNumCases()));

    std::string SwVar = getNextVar();
    // No need to save temporary var into symTable.
    OS << formatv("auto {0} = {1}.{2};\n", SwVar, Builder, Call);

    for (auto C : SwI->cases()) {
      OS << formatv("{0}->addCase({1}, {2});\n", SwVar, asStr(C.getCaseValue()),
                    asStr(C.getCaseSuccessor()));
    }
    return;
  } break;
  case Instruction::IndirectBr: {
    auto *InbrI = cast<IndirectBrInst>(I);
    Call =
        formatv("CreateIndirectBr({0}, {1})", Op1, InbrI->getNumDestinations());
    std::string InbrVar = getNextVar();
    OS << formatv("auto {0} = {1}.{2};\n", InbrVar, Builder, Call);

    for (auto C : InbrI->successors()) {
      OS << InbrVar << "->addDestination(" << asStr(C) << ");\n";
    }
  } break;
  case Instruction::Invoke: {
    auto *InvI = dyn_cast<InvokeInst>(I);
    SmallVector<std::string> Tmp;
    auto Args =
        "{" +
        formatv("{0}", (llvm::transform(InvI->args(), back_inserter(Tmp),
                                        [&](Value *V) { return asStr(V); }),
                        make_range(Tmp.begin(), Tmp.end()))) +
        "}";
    std::string FunDecl = getNextVar();
    OS << formatv("auto {0} = {1}->getOrInsertFunction(\"{2}\", {3});\n",
                  FunDecl, ModName, I->getOperand(2)->getName(),
                  asStr(InvI->getFunctionType()));
    Call = formatv("CreateInvoke({0}, {1}, {2}, {3})", FunDecl,
                   asStr(InvI->getNormalDest()), asStr(InvI->getUnwindDest()),
                   Args);
    // TODO: Handle operand bundles.
  } break;
  case Instruction::Resume: {
    Call = formatv("CreateResume({0})", Op1);
  } break;
  case Instruction::Unreachable: {
    Call = "CreateUnreachable()";
  } break;
  case Instruction::CleanupRet: {
    auto *CurI = cast<CleanupReturnInst>(I);
    Call = formatv("CreateCleanupRet({0}, {1})", Op1,
                   asStr(CurI->getUnwindDest()));
  } break;
  case Instruction::CatchRet: {
    Call = formatv("CreateCatchRet({0}, {1})", Op1, Op2);
  } break;
  case Instruction::CatchSwitch: {
    auto *SwI = cast<CatchSwitchInst>(I);
    Call = formatv("CreateCatchSwitch({0}, {1}, {2})", Op1, Op2,
                   std::to_string(SwI->getNumHandlers()));

    std::string SwVar = asStr(SwI);
    OS << formatv("auto {0} = {1}.{2};\n", SwVar, Builder, Call);
    for (auto C : SwI->handlers()) {
      OS << formatv("{0}->addHandler({1});\n", SwVar, asStr(C));
    }
    return;
  } break;
  case Instruction::CallBr: {
    auto *CallBRI = cast<CallBrInst>(I);
    SmallVector<std::string> Tmp;
    auto Inddest =
        "{" +
        formatv("{0}", (llvm::transform(CallBRI->getIndirectDests(),
                                        back_inserter(Tmp),
                                        [&](Value *V) { return asStr(V); }),
                        make_range(Tmp.begin(), Tmp.end()))) +
        "}";

    SmallVector<std::string> Tmp2;
    auto Args =
        "{" +
        formatv("{0}", (llvm::transform(CallBRI->args(), back_inserter(Tmp2),
                                        [&](Value *V) { return asStr(V); }),
                        make_range(Tmp2.begin(), Tmp2.end()))) +
        "}";
    Call = formatv("CreateCallBr({0}, {1}, {2}, {3}, {4})",
                   asStr(CallBRI->getFunctionType()),
                   asStr(I->getOperand(I->getNumOperands() - 1)),
                   asStr(CallBRI->getDefaultDest()), Inddest, Args);
    // TODO: Handle operand bundles.
  } break;
  case Instruction::FNeg: {
    Call = formatv("CreateFNeg({0})", Op1);
    // TODO: Handle FPMathTag.
  } break;
  case Instruction::Add: {
    Call = getBinArithOp("Add", Op1, Op2, I);
  } break;
  case Instruction::FAdd: {
    Call = getFPBinArithOp("FAdd", Op1, Op2, I);
  } break;
  case Instruction::Sub: {
    Call = getBinArithOp("Sub", Op1, Op2, I);
  } break;
  case Instruction::FSub: {
    Call = getFPBinArithOp("FSub", Op1, Op2, I);
  } break;
  case Instruction::Mul: {
    Call = getBinArithOp("Mul", Op1, Op2, I);
  } break;
  case Instruction::FMul: {
    Call = getFPBinArithOp("FMul", Op1, Op2, I);
  } break;
  case Instruction::UDiv: {
    Call = formatv("CreateUDiv({0}, {1}, \"\", {2})", Op1, Op2,
                   toStr(I->isExact()));
  } break;
  case Instruction::SDiv: {
    Call = formatv("CreateSDiv({0}, {1}, \"\", {2})", Op1, Op2,
                   toStr(I->isExact()));
  } break;
  case Instruction::FDiv: {
    Call = getFPBinArithOp("FDiv", Op1, Op2, I);
  } break;
  case Instruction::URem: {
    Call = formatv("CreateURem({0}, {1})", Op1, Op2);
  } break;
  case Instruction::SRem: {
    Call = formatv("CreateSRem({0}, {1})", Op1, Op2);
  } break;
  case Instruction::FRem: {
    Call = getFPBinArithOp("FRem", Op1, Op2, I);
  } break;
  case Instruction::Shl: {
    Call = getBinArithOp("Shl", Op1, Op2, I);
  } break;
  case Instruction::LShr: {
    Call = formatv("CreateLShr({0}, {1}, \"\", {2})", Op1, Op2,
                   toStr(I->isExact()));
  } break;
  case Instruction::AShr: {
    Call = formatv("CreateAShr({0}, {1}, \"\", {2})", Op1, Op2,
                   toStr(I->isExact()));
  } break;
  case Instruction::And: {
    Call = formatv("CreateAnd({0}, {1})", Op1, Op2);
  } break;
  case Instruction::Or: {
    Call = formatv("CreateOr({0}, {1})", Op1, Op2);
  } break;
  case Instruction::Xor: {
    Call = formatv("CreateXor({0}, {1})", Op1, Op2);
  } break;
  case Instruction::Alloca: {
    auto *AlI = cast<AllocaInst>(I);
    auto Val = AlI->getArraySize();
    auto ValStr = Val ? asStr(Val) : "nullptr";
    Call =
        formatv("CreateAlloca({0}, {1}, {2})", asStr(AlI->getAllocatedType()),
                AlI->getAddressSpace(), ValStr);
  } break;
  case Instruction::Load: {
    auto *LI = cast<LoadInst>(I);
    Call = formatv("CreateLoad({0}, {1}, {2})", asStr(I->getType()), Op1,
                   toStr(LI->isVolatile()));
  } break;
  case Instruction::Store: {
    auto *SI = dyn_cast<StoreInst>(I);
    Call = formatv("CreateStore({0}, {1}, {2})", Op1, Op2,
                   toStr(SI->isVolatile()));
  } break;
  case Instruction::GetElementPtr: {
    auto *GEPI = dyn_cast<GetElementPtrInst>(I);
    std::string StrList = formatv("{0}ArrayRef<{0}Value*>(", LLVMPrefix);
    if (I->getNumOperands() > 2)
      StrList += "{";
    SmallVector<std::string> Tmp;
    iterator_range<const Use *> GepRange(I->op_begin() + 1, I->op_end());
    StrList +=
        formatv("{0}", (llvm::transform(GepRange, back_inserter(Tmp),
                                        [&](Value *V) { return asStr(V); }),
                        make_range(Tmp.begin(), Tmp.end())));
    if (I->getNumOperands() > 2)
      StrList += "}";
    StrList += ")";
    Call = formatv("CreateGEP({0}, {1}, {2}, \"\", {3})",
                   asStr(GEPI->getSourceElementType()), Op1, StrList,
                   toStr(GEPI->isInBounds()));
  } break;
  case Instruction::Fence: {
    auto *FI = dyn_cast<FenceInst>(I);
    Call = formatv("CreateFence({0}, {1})", asStr(FI->getOrdering()),
                   asStr(FI->getSyncScopeID()));
  } break;
  case Instruction::AtomicCmpXchg: {
    auto *AcmpxI = dyn_cast<AtomicCmpXchgInst>(I);
    Call = formatv(
        "CreateAtomicCmpXchg({0}, {1}, {2}, Align({3}), {4}, {5}, {6})", Op1,
        Op2, Op3, AcmpxI->getAlign().value(),
        asStr(AcmpxI->getSuccessOrdering()),
        asStr(AcmpxI->getFailureOrdering()), asStr(AcmpxI->getSyncScopeID()));
  } break;
  case Instruction::AtomicRMW: {
    auto *ArmwI = dyn_cast<AtomicRMWInst>(I);
    Call = formatv("CreateAtomicRMW({0}, {1}, {2}, Align({3}), {4}, {5})",
                   asStr(ArmwI->getOperation()), Op1, Op2,
                   ArmwI->getAlign().value(), asStr(ArmwI->getOrdering()),
                   asStr(ArmwI->getSyncScopeID()));
  } break;
  case Instruction::Trunc: {
    Call = formatv("CreateTrunc({0}, {1})", Op1, asStr(I->getType()));
  } break;
  case Instruction::ZExt: {
    Call = formatv("CreateZExt({0}, {1}, \"\", {2})", Op1, asStr(I->getType()),
                   toStr(I->hasNonNeg()));
  } break;
  case Instruction::SExt: {
    Call = formatv("CreateSExt({0}, {1})", Op1, asStr(I->getType()));
  } break;
  case Instruction::FPToUI: {
    Call = formatv("CreateFPToUI({0}, {1})", Op1, asStr(I->getType()));
  } break;
  case Instruction::FPToSI: {
    Call = formatv("CreateFPToSI({0}, {1})", Op1, asStr(I->getType()));
  } break;
  case Instruction::UIToFP: {
    Call = formatv("CreateUIToFP({0}, {1})", Op1, asStr(I->getType()));
  } break;
  case Instruction::SIToFP: {
    Call = formatv("CreateSIToFP({0}, {1})", Op1, asStr(I->getType()));
  } break;
  case Instruction::FPTrunc: {
    Call = formatv("CreateFPTrunc({0}, {1})", Op1, asStr(I->getType()));
  } break;
  case Instruction::FPExt: {
    Call = formatv("CreateFPExt({0}, {1})", Op1, asStr(I->getType()));
  } break;
  case Instruction::PtrToInt: {
    Call = formatv("CreatePtrToInt({0}, {1})", Op1, asStr(I->getType()));
  } break;
  case Instruction::IntToPtr: {
    Call = formatv("CreateIntToPtr({0}, {1})", Op1, asStr(I->getType()));
  } break;
  case Instruction::BitCast: {
    Call = formatv("CreateBitCast({0}, {1})", Op1, asStr(I->getType()));
  } break;
  case Instruction::AddrSpaceCast: {
    Call = formatv("CreateAddrSpaceCast({0}, {1})", Op1, asStr(I->getType()));
  } break;
  case Instruction::CleanupPad: {
    auto *CupI = dyn_cast<CleanupPadInst>(I);
    SmallVector<std::string> Tmp;
    auto ArgsStr =
        "{" +
        formatv("{0}",
                (llvm::transform(CupI->arg_operands(), back_inserter(Tmp),
                                 [&](Value *V) { return asStr(V); }),
                 make_range(Tmp.begin(), Tmp.end()))) +
        "}";
    Call = formatv("CreateCleanupPad({0}, {1})", Op1, ArgsStr);
  } break;
  case Instruction::CatchPad: {
    auto *CapI = dyn_cast<CatchPadInst>(I);
    SmallVector<std::string> Tmp;
    auto ArgsStr =
        "{" +
        formatv("{0}",
                (llvm::transform(CapI->arg_operands(), back_inserter(Tmp),
                                 [&](Value *V) { return asStr(V); }),
                 make_range(Tmp.begin(), Tmp.end()))) +
        "}";
    Call = formatv("CreateCatchPad({0}, {1})", Op1, ArgsStr);
  } break;
  case Instruction::ICmp: {
    auto *CmpI = dyn_cast<CmpInst>(I);
    std::string CmpPredicate = LLVMPrefix + asStr(CmpI->getPredicate());
    Call = formatv("CreateICmp({0}, {1}, {2})", CmpPredicate, Op1, Op2);
  } break;
  case Instruction::FCmp: {
    auto *CmpI = dyn_cast<CmpInst>(I);
    std::string CmpPredicate = LLVMPrefix + asStr(CmpI->getPredicate());
    Call = formatv("CreateFCmp({0}, {1}, {2})", CmpPredicate, Op1, Op2);
  } break;
  case Instruction::PHI: {
    auto *PhI = dyn_cast<PHINode>(I);
    std::string PhVar = asStr(PhI);
    OS << formatv("auto {0} = {1}.CreatePHI({2}, {3});\n", PhVar, Builder,
                  asStr(I->getType()), PhI->getNumIncomingValues());

    unsigned I = 0;
    for (auto B : PhI->blocks()) {
      auto IncVal = asStr(PhI->getIncomingValue(I));
      // Phis might contain incoming not yet defined variable in such case
      // we save the line we would output here and output it after the
      // whole function body was outputted.
      std::string Line =
          formatv("{0}->addIncoming({1}, {2});\n", PhVar, IncVal, asStr(B));
      PhiIncomings.push_back(Line);
      ++I;
    }
    return;
  } break;
  case Instruction::Call: {
    auto *FcI = dyn_cast<CallBase>(I);
    std::string ArgDecl = "";
    if (FcI->arg_size() > 0) {
      ArgDecl = getNextVar();
      OS << "Value *" << ArgDecl << "[] = {";
      SmallVector<std::string> Tmp;
      OS << formatv("{0}", (llvm::transform(FcI->args(), back_inserter(Tmp),
                                            [&](Value *V) { return asStr(V); }),
                            make_range(Tmp.begin(), Tmp.end())));
      OS << "};\n";
    }

    /* TODO: Implement.
    std::string bundleDecl = "";
    if (FcI->hasOperandBundles()) {
        bundleDecl = getNextVar();
        OS << "Value *" << bundleDecl << "[] = {";
        IsFirst = true;
        for(unsigned I = 0; I < FcI->getNumOperandBundles(); ++I) {
            if(!IsFirst) OS << ", ";
            // TODO: Probably create OperandBundleUse from the current gotten
            // from getOperandBundleAt and cast construct OperandBundleDefT with
            // it, but this requires conversion of Inputs which are Use.
            OS << "...";
            IsFirst = false;
        }
        OS << "};\n";
    }
    */

    auto *Fun = dyn_cast<Function>(I->getOperand(I->getNumOperands() - 1));
    std::string FunDecl = getNextVar();
    if (!Fun)
      Fun = FcI->getCalledFunction();

    if (!Fun) {
      if (auto *InA = dyn_cast<InlineAsm>(FcI->getCalledOperand())) {
        if (ArgDecl.empty()) {
          Call = formatv("CreateCall({0}, {1})", asStr(InA->getFunctionType()),
                         asStr(I->getOperand(I->getNumOperands() - 1)));
        } else {
          Call = formatv(
              "CreateCall({0}, {1}, {2})", asStr(InA->getFunctionType()),
              asStr(I->getOperand(I->getNumOperands() - 1)), ArgDecl);
        }
      } else if (FcI->isIndirectCall()) {
        if (ArgDecl.empty()) {
          Call = formatv("CreateCall({0}, {1})", asStr(FcI->getFunctionType()),
                         asStr(I->getOperand(I->getNumOperands() - 1)));
        } else {
          Call = formatv(
              "CreateCall({0}, {1}, {2})", asStr(FcI->getFunctionType()),
              asStr(I->getOperand(I->getNumOperands() - 1)), ArgDecl);
        }
      } else if (auto *GA = dyn_cast<GlobalAlias>(FcI->getCalledOperand())) {
        Fun = dyn_cast<Function>(GA->getAliaseeObject());
        assert(Fun && "Global alias is not a function");
        OS << formatv("auto {0} = {1}->getOrInsertFunction(\"{2}\", {3});\n",
                      FunDecl, ModName, Fun->getName(),
                      asStr(Fun->getFunctionType()));
        if (!ArgDecl.empty())
          Call = formatv("CreateCall({0}, {1})", FunDecl, ArgDecl);
        else
          Call = formatv("CreateCall({0})", FunDecl);
      } else {
        OS << "/* TODO: Unknown CallBase type in Call instruction */\n";
        return;
      }
    } else {
      OS << formatv("auto {0} = {1}->getOrInsertFunction(\"{2}\", {3});\n",
                    FunDecl, ModName, Fun->getName(),
                    asStr(Fun->getFunctionType()));
      if (!ArgDecl.empty())
        Call = formatv("CreateCall({0}, {1})", FunDecl, ArgDecl);
      else
        Call = formatv("CreateCall({0})", FunDecl);
    }
  } break;
  case Instruction::Select: {
    Call = formatv("CreateSelect({0}, {1}, {2})", Op1, Op2, Op3);
  } break;
  case Instruction::UserOp1: {
    // Internal opcode.
    OS << "// TODO: UserOp1 appeared in the IR\n";
    return;
  };
  case Instruction::UserOp2: {
    // Internal opcode.
    OS << "// TODO: UserOp2 appeared in the IR\n";
    return;
  };
  case Instruction::VAArg: {
    Call = formatv("CreateVAArg({0}, {1})", Op1, asStr(I->getType()));
  } break;
  case Instruction::ExtractElement: {
    Call = formatv("CreateExtractElement({0}, {1})", Op1, Op2);
  } break;
  case Instruction::InsertElement: {
    Call = formatv("CreateInsertElement({0}, {1}, {2})", Op1, Op2, Op3);
  } break;
  case Instruction::ShuffleVector: {
    auto *SvI = cast<ShuffleVectorInst>(I);
    SmallVector<std::string> Tmp;
    auto MaskStr =
        "{" +
        formatv("{0}",
                (llvm::transform(SvI->getShuffleMask(), back_inserter(Tmp),
                                 [&](int V) { return std::to_string(V); }),
                 make_range(Tmp.begin(), Tmp.end()))) +
        "}";
    Call = formatv("CreateShuffleVector({0}, {1}, {2})", Op1, Op2, MaskStr);
  } break;
  case Instruction::ExtractValue: {
    auto *EvI = cast<ExtractValueInst>(I);
    SmallVector<std::string> Tmp;
    auto ArgStr =
        "{" +
        formatv("{0}",
                (llvm::transform(EvI->getIndices(), back_inserter(Tmp),
                                 [&](int V) { return std::to_string(V); }),
                 make_range(Tmp.begin(), Tmp.end()))) +
        "}";
    Call = formatv("CreateExtractValue({0}, {1})", Op1, ArgStr);
  } break;
  case Instruction::InsertValue: {
    auto *IvI = cast<InsertValueInst>(I);
    SmallVector<std::string> Tmp;
    auto ArgStr =
        "{" +
        formatv("{0}",
                (llvm::transform(IvI->getIndices(), back_inserter(Tmp),
                                 [&](int V) { return std::to_string(V); }),
                 make_range(Tmp.begin(), Tmp.end()))) +
        "}";
    Call = formatv("CreateInsertValue({0}, {1}, {2})", Op1, Op2, ArgStr);
  } break;
  case Instruction::LandingPad: {
    auto *LPI = dyn_cast<LandingPadInst>(I);
    std::string LPVar = asStr(I);
    OS << formatv("auto {0} = {1}.CreateLandingPad({2}, {3});\n", LPVar,
                  Builder, asStr(I->getType()), LPI->getNumClauses());
    for (unsigned I = 0; I < LPI->getNumClauses(); ++I) {
      OS << formatv("{0}->addClause({1});\n", LPVar, asStr(LPI->getClause(I)));
    }
    return;
  } break;
  case Instruction::Freeze: {
    Call = formatv("CreateFreeze({0})", Op1);
  } break;
  default:
    OS << "// Unknown instruction: " << *I << "\n";
    return;
  }

  // Check if Call returns a value.
  if (!I->getType()->isVoidTy())
    OS << "auto " << asStr(I) << " = ";
  OS << Builder << "." << Call << ";\n";
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  cl::HideUnrelatedOptions(Ir2BCat);
  cl::ParseCommandLineOptions(argc, argv);

  // Parse input file.
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

  // Output generation.
  IR2Builder IR2B;
  if (!OutputFilename.empty()) {
    std::error_code EC;
    std::unique_ptr<ToolOutputFile> Out(
        new ToolOutputFile(OutputFilename, EC, sys::fs::OF_None));
    if (EC) {
      errs() << EC.message() << '\n';
      exit(1);
    }
    IR2B.convert(*(M.get()), Out->os());
    Out->keep();
  } else {
    IR2B.convert(*(M.get()), outs());
  }

  return 0;
}
