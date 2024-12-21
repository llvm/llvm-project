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

  inline bool hasName(const Value *Op) {
    return !isa<Constant>(Op) && !isa<InlineAsm>(Op);
  }

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

std::string IR2Builder::getNextVar() { return "v0" + std::to_string(VarI++); }

static std::string toStr(bool B) { return B ? "true" : "false"; }

static std::string escape(std::string S) {
  std::string Tmp;
  raw_string_ostream OS(Tmp);
  printEscapedString(S, OS);
  return Tmp;
}

static std::string sanitize(std::string S) {
  std::stringstream SS;
  for (size_t I = 0; I < S.size(); ++I) {
    if (!std::isalnum(S[I]))
      SS << "_" << static_cast<unsigned>(S[I]) << "_";
    else
      SS << S[I];
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
  if (auto IT = dyn_cast<IntegerType>(T)) {
    switch (IT->getBitWidth()) {
    case 8:
    case 16:
    case 32:
    case 64:
      TCall = ("getInt" + Twine(IT->getBitWidth()) + "Ty()").str();
      break;
    default:
      TCall = "getIntNTy(" + std::to_string(IT->getBitWidth()) + ")";
      break;
    }
  } else if (T->isVoidTy())
    TCall = "getVoidTy()";
  else if (T->isFloatTy())
    TCall = "getFloatTy()";
  else if (T->isDoubleTy())
    TCall = "getDoubleTy()";
  else if (auto ST = dyn_cast<StructType>(T)) {
    std::string Elements = LLVMPrefix + "ArrayRef<Type *>(";
    if (ST->getNumElements() > 1)
      Elements += "{";
    bool First = true;
    for (auto E : ST->elements()) {
      if (!First)
        Elements += ", ";
      Elements += asStr(E);
      First = false;
    }
    if (ST->getNumElements() > 1)
      Elements += "}";
    Elements += ")";
    return LLVMPrefix + "StructType::create(" + Ctx + ", " + Elements + ")";
  } else if (auto AT = dyn_cast<ArrayType>(T)) {
    return LLVMPrefix + "ArrayType::get(" + asStr(AT->getElementType()) +
           ", " + std::to_string(AT->getArrayNumElements()) + ")";
  } else if (T->isBFloatTy())
    TCall = "getBFloatTy()";
  else if (auto VT = dyn_cast<VectorType>(T)) {
    std::string ElemCount =
        LLVMPrefix + "ElementCount::get(" +
        std::to_string(VT->getElementCount().getKnownMinValue()) + ", " +
        toStr(VT->getElementCount().isScalable()) + ")";
    return LLVMPrefix + "VectorType::get(" + asStr(VT->getElementType()) +
           ", " + ElemCount + ")";
  } else if (auto FT = dyn_cast<FunctionType>(T)) {
    TCall = LLVMPrefix + "FunctionType::get(" + asStr(FT->getReturnType()) +
            ", {";

    bool IsFirst = true;
    for (auto A : FT->params()) {
      if (!IsFirst)
        TCall += ", ";
      TCall += asStr(A);
      IsFirst = false;
    }
    TCall += "}, " + toStr(FT->isVarArg()) + ")";
    return TCall;
  } else if (T->isPointerTy())
    TCall = "getPtrTy()";
  else if (T->isHalfTy())
    return LLVMPrefix + "Type::getHalfTy(" + Ctx + ")";
  else if (T->isBFloatTy())
    return LLVMPrefix + "Type::getBFloatTy(" + Ctx + ")";
  else if (T->isX86_FP80Ty())
    return LLVMPrefix + "Type::getX86_FP80Ty(" + Ctx + ")";
  else if (T->isFP128Ty())
    return LLVMPrefix + "Type::getFP128Ty(" + Ctx + ")";
  else if (T->isPPC_FP128Ty())
    return LLVMPrefix + "Type::getPPC_FP128Ty(" + Ctx + ")";
  else if (T->isX86_AMXTy())
    return LLVMPrefix + "Type::getX86_AMXTy(" + Ctx + ")";
  else if (T->isLabelTy())
    return LLVMPrefix + "Type::getLabelTy(" + Ctx + ")";
  else if (T->isMetadataTy())
    return LLVMPrefix + "Type::getMetadataTy(" + Ctx + ")";
  else if (T->isTokenTy())
    return LLVMPrefix + "Type::getTokenTy(" + Ctx + ")";
  else
    return "/* TODO: Unknown type */";

  return Builder + "." + TCall;
}

std::string IR2Builder::asStr(const Constant *C) {
  if (auto CI = dyn_cast<ConstantInt>(C)) {
    // TODO: Sign has to be determined.
    auto CVal = CI->getValue();
    return LLVMPrefix + "ConstantInt::get(" + asStr(C->getType()) + ", " +
           std::to_string(CVal.getSExtValue()) + ")";
  }
  if (auto CF = dyn_cast<ConstantFP>(C)) {
    auto CVal = CF->getValue();
    double DVal = CVal.convertToDouble();
    std::string Val = std::to_string(DVal);
    if (std::isnan(DVal) || std::isinf(DVal))
      Val = "\"" + Val + "\"";
    // TODO: Handle double to string conversion to include all digits.
    return LLVMPrefix + "ConstantFP::get(" + asStr(C->getType()) + ", " +
           Val + ")";
  }
  if (auto AT = dyn_cast<ConstantAggregate>(C)) {
    std::string Values;
    bool First = true;
    for (unsigned I = 0; I < AT->getNumOperands(); ++I) {
      if (!First)
        Values += ", ";
      Values += asStr(AT->getOperand(I));
      First = false;
    }

    std::string ClassName;
    if (isa<ConstantArray>(C)) {
      ClassName = "ConstantArray";
      Values = LLVMPrefix + "ArrayRef<" + LLVMPrefix + "Constant *>(" +
               (AT->getNumOperands() > 1 ? std::string("{") : std::string("")) +
               Values +
               (AT->getNumOperands() > 1 ? std::string("}") : std::string("")) +
               ")";
    } else if (isa<ConstantStruct>(C))
      ClassName = "ConstantStruct";
    else if (isa<ConstantVector>(C)) {
      Values = "{" + Values + "}";
      // ConstantVector does not take type as 1st arg.
      return LLVMPrefix + "ConstantVector::get(" + Values + ")";
    } else
      return "/* TODO: Unknown aggregate constant */";

    return LLVMPrefix + ClassName + "::get(" + asStr(C->getType()) + ", " +
           Values + ")";
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
      Values = "";
      bool First = true;
      for (auto a : CDS->getAsString().str()) {
        if (First) {
          Values += std::to_string(static_cast<uint8_t>(a));
          First = false;
        } else {
          Values += ", " + std::to_string(static_cast<uint8_t>(a));
        }
      }
      return LLVMPrefix + ClassName + "::get(" + Ctx + ", " + LLVMPrefix +
             "ArrayRef<uint8_t>({" + Values + "}))";
    }
    if (CDS->isCString()) {
      Values = "";
      bool First = true;
      for (auto A : CDS->getAsCString().str()) {
        if (First) {
          Values += std::to_string(static_cast<uint8_t>(A));
          First = false;
        } else {
          Values += ", " + std::to_string(static_cast<uint8_t>(A));
        }
      }
      return LLVMPrefix + ClassName + "::get(" + Ctx + ", " + LLVMPrefix +
             "ArrayRef<uint8_t>({" + Values + "}))";
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
      Values = LLVMPrefix + "ArrayRef<" + ElemTy + ">(";
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

    return LLVMPrefix + ClassName + "::get(" + Ctx + ", " + Values + ")";
  }
  if (isa<ConstantAggregateZero>(C)) {
    return LLVMPrefix + "ConstantAggregateZero::get(" + asStr(C->getType()) +
           ")";
  }
  if (isa<PoisonValue>(C)) {
    return LLVMPrefix + "PoisonValue::get(" + asStr(C->getType()) + ")";
  }
  if (isa<UndefValue>(C)) {
    return LLVMPrefix + "UndefValue::get(" + asStr(C->getType()) + ")";
  }
  if (auto ba = dyn_cast<BlockAddress>(C)) {
    return LLVMPrefix + "BlockAddress::get(" + asStr(ba->getFunction()) +
           ", " + asStr(ba->getBasicBlock()) + ")";
  }
  if (isa<ConstantPointerNull>(C)) {
    return LLVMPrefix + "ConstantPointerNull::get(" + asStr(C->getType()) +
           ")";
  }
  if (auto CTN = dyn_cast<ConstantTargetNone>(C)) {
    auto CTNType = CTN->getType();

    std::string TypeStr = "{";
    bool First = true;
    for (unsigned I = 0; I < CTNType->getNumTypeParameters(); ++I) {
      if (!First)
        TypeStr += ", ";
      TypeStr += asStr(CTNType->getTypeParameter(I));
      First = false;
    }
    TypeStr += "}";

    std::string IntsStr = "{";
    First = true;
    for (unsigned I = 0; I < CTNType->getNumIntParameters(); ++I) {
      if (!First)
        IntsStr += ", ";
      IntsStr += std::to_string(CTNType->getIntParameter(I));
      First = false;
    }
    IntsStr += "}";

    std::string CTNName = "\"" + escape(CTNType->getName().str()) + "\"";
    std::string TET = LLVMPrefix + "TargetExtType::get(" + Ctx + ", " +
                      CTNName + ", " + TypeStr + ", " + IntsStr + ")";

    return LLVMPrefix + "ConstantTargetNone::get(" + TET + ")";
  }
  if (isa<ConstantTokenNone>(C)) {
    return LLVMPrefix + "ConstantTokenNone::get(" + Ctx + ")";
  }
  if (auto CE = dyn_cast<ConstantExpr>(C)) {
    (void)CE;
    return "/* TODO: ConstantExpr creation */";
    // TODO: Dunno how to create this... Fails either on out of range or
    // even on incorrect opcode.
    // return LLVMPrefix + "ConstantExpr::get(" +
    // std::to_string(CE->getOpcode()) + ", " +
    //       asStr(CE->getOperand(0)) + ", " + asStr(CE->getOperand(1)) + ")";
  }
  if (auto CE = dyn_cast<ConstantPtrAuth>(C)) {
    (void)CE;
    return "/* TODO: ConstantPtrAuth value creation */";
  }
  if (auto CE = dyn_cast<DSOLocalEquivalent>(C)) {
    (void)CE;
    return "/* TODO: DSOLocalEquivalent value creation */";
  }
  if (auto CE = dyn_cast<NoCFIValue>(C)) {
    (void)CE;
    return "/* TODO: NoCFIValue value creation */";
  }
  if (auto CE = dyn_cast<GlobalValue>(C)) {
    // This should not really happen as asStr for Value should be always called.
    return asStr(dyn_cast<Value>(CE));
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

  return LLVMPrefix + "InlineAsm::get(" + asStr(Op->getFunctionType()) +
         ", " + "\"" + escape(Op->getAsmString()) + "\", " + "\"" +
         escape(Op->getConstraintString()) + "\", " +
         toStr(Op->hasSideEffects()) + ", " + toStr(Op->isAlignStack()) +
         ", " + GetAsmDialect(Op->getDialect()) + ", " +
         toStr(Op->canThrow()) + ")";
}

std::string IR2Builder::asStr(const Metadata *Op) {
  if (auto MDN = dyn_cast<MDNode>(Op)) {
    std::string Args = "{";
    bool First = true;
    for (unsigned I = 0; I < MDN->getNumOperands(); ++I) {
      if (!First)
        Args += ", ";
      Args += asStr(MDN->getOperand(I));
      First = false;
    }
    Args += "}";
    return LLVMPrefix + "MDNode::get(" + Ctx + ", " + Args + ")";
  }
  if (auto VAM = dyn_cast<ValueAsMetadata>(Op)) {
    return LLVMPrefix + "ValueAsMetadata::get(" + asStr(VAM->getValue()) + ")";
  }
  if (auto MDS = dyn_cast<MDString>(Op)) {
    return LLVMPrefix + "MDString::get(" + Ctx + ", \"" +
           escape(MDS->getString().str()) + "\")";
  } 
  
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
    return asStr(dyn_cast<Constant>(Op));
  if (auto InA = dyn_cast<InlineAsm>(Op))
    return asStr(InA);
  if (auto Mtd = dyn_cast<MetadataAsValue>(Op)) {
    return LLVMPrefix + "MetadataAsValue::get(" + Ctx + ", " +
           asStr(Mtd->getMetadata()) + ")";
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
  return "Create" + Name + "(" + Op1 + ", " + Op2 + ", \"\", " +
         toStr(I->hasNoUnsignedWrap()) + ", " + toStr(I->hasNoSignedWrap()) +
         ")";
}

std::string getFPBinArithOp(std::string Name, std::string Op1, std::string Op2,
                            const Instruction *I) {
  return "Create" + Name + "(" + Op1 + ", " + Op2 + ")";
  // TODO: Handle FPMathTag.
}

std::string IR2Builder::asStr(ConstantRange &CR) {
  std::stringstream SS;
  auto NumBitsStr = std::to_string(CR.getBitWidth());
  auto Lower = std::to_string(CR.getLower().getLimitedValue());
  auto Upper = std::to_string(CR.getUpper().getLimitedValue());
  SS << LLVMPrefix << "ConstantRange(APInt(" << NumBitsStr << ", " << Lower
     << ", true), "
     << "APInt(" << NumBitsStr << ", " << Upper << ", true))";
  return SS.str();
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
    std::string Args = "{";
    bool First = true;
    for (auto CR : Att.getValueAsConstantRangeList()) {
      if (!First)
        Args += ", ";
      Args += asStr(CR);
      First = false;
    }
    Args += "}";
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
        if (auto BA = dyn_cast<BlockAddress>(G.getInitializer())) {
          OS << "auto " << asStr(BA->getFunction()) << " = " << ModName
             << "->getOrInsertFunction(\"" << BA->getFunction()->getName()
             << "\", " << asStr(BA->getFunction()->getFunctionType())
             << ");\n";
          // Extract basic block.
          // TODO: The function might need to be created in here
          // since the basic blocks don't exist yet.
          OS << "// TODO: Basic block extraction\n";
        }
        InitVal = asStr(G.getInitializer());
      }

      std::string GName = asStr(&G);
      OS << "auto " << GName << " = new " << LLVMPrefix << "GlobalVariable(*"
         << ModName << ", " << asStr(G.getValueType()) << ", "
         << toStr(G.isConstant()) << ", " << asStr(G.getLinkage()) << ", " << InitVal
         << ", \"" << escape(G.getName().str()) << "\", "
         << "nullptr"
         << ", " << asStr(G.getThreadLocalMode()) << ", "
         << std::to_string(G.getAddressSpace()) << ", "
         << toStr(G.isExternallyInitialized()) << ");\n";
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
  OS << LLVMPrefix << "Function *" << FDecl;
  OS << " = " << LLVMPrefix << "Function::Create("
     << asStr(F.getFunctionType()) << ", " << asStr(F.getLinkage()) << ", \""
     << F.getName() << "\", " << ModName << ");\n";

  OS << "\n";

  // Set attributes.
  if (F.getCallingConv() != CallingConv::C) { // C is default.
    OS << FDecl << "->setCallingConv(" + asStr(F.getCallingConv())
       << ");\n";
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
    OS << "auto " << asStr(&Arg) << " = " << FDecl << "->getArg("
       << std::to_string(I) << ");\n";
  }

  if (UseRPO) {
    ReversePostOrderTraversal<Function *> RPOT(&F);

    // Basic block declaration in order.
    for (BasicBlock *BB : RPOT) {
      std::string bbName = asStr(BB);
      OS << LLVMPrefix << "BasicBlock* " << bbName << " = " << LLVMPrefix
         << "BasicBlock::Create(" << Ctx << ", \"" << BB->getName() << "\", "
         << FDecl << ");\n";
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
      std::string bbName = asStr(&BB);
      OS << LLVMPrefix << "BasicBlock* " << bbName << " = " << LLVMPrefix
         << "BasicBlock::Create(" << Ctx << ", \"" << BB.getName() << "\", "
         << FDecl << ");\n";
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
    const BranchInst *BI = dyn_cast<BranchInst>(I);
    if (BI->isUnconditional()) {
      Call = "CreateBr(" + Op1 + ")";
    } else {
      Call = "CreateCondBr(" + Op1 + ", " + Op3 + ", " + Op2 + ")";
    }
  } break;
  case Instruction::Switch: {
    auto SwI = dyn_cast<SwitchInst>(I);
    Call = "CreateSwitch(" + Op1 + ", " + Op2 + ", " +
           std::to_string(SwI->getNumCases()) + ")";

    std::string SwVar = getNextVar();
    // No need to save temporary var into symTable.
    OS << "auto " << SwVar << " = " << Builder << "." << Call << ";\n";

    for (auto C : SwI->cases()) {
      OS << SwVar << "->addCase(" << asStr(C.getCaseValue()) << ", "
         << asStr(C.getCaseSuccessor()) << ");\n";
    }
    return;
  } break;
  case Instruction::IndirectBr: {
    auto InbrI = dyn_cast<IndirectBrInst>(I);
    Call = "CreateIndirectBr(" + Op1 + ", " +
           std::to_string(InbrI->getNumDestinations()) + ")";
    std::string InbrVar = getNextVar();
    OS << "auto " << InbrVar << " = " << Builder << "." << Call << ";\n";

    for (auto C : InbrI->successors()) {
      OS << InbrVar << "->addDestination(" << asStr(C) << ");\n";
    }
  } break;
  case Instruction::Invoke: {
    auto InvI = dyn_cast<InvokeInst>(I);
    std::string Args = "{";
    bool First = true;
    for (unsigned I = 0; I < InvI->arg_size(); ++I) {
      if (!First)
        Args += ", ";
      Args += asStr(InvI->getArgOperand(I));
      First = false;
    }
    Args += "}";
    std::string FunDecl = getNextVar();
    OS << "auto " << FunDecl << " = " << ModName << "->getOrInsertFunction(\""
       << I->getOperand(2)->getName() << "\", "
       << asStr(InvI->getFunctionType()) << ");\n";
    Call = "CreateInvoke(" + FunDecl + ", " + asStr(InvI->getNormalDest()) +
           ", " + asStr(InvI->getUnwindDest()) + ", " + Args + ")";
    // TODO: Handle operand bundles.
  } break;
  case Instruction::Resume: {
    Call = "CreateResume(" + Op1 + ")";
  } break;
  case Instruction::Unreachable: {
    Call = "CreateUnreachable()";
  } break;
  case Instruction::CleanupRet: {
    auto CurI = dyn_cast<CleanupReturnInst>(I);
    Call =
        "CreateCleanupRet(" + Op1 + ", " + asStr(CurI->getUnwindDest()) + ")";
  } break;
  case Instruction::CatchRet: {
    Call = "CreateCatchRet(" + Op1 + ", " + Op2 + ")";
  } break;
  case Instruction::CatchSwitch: {
    auto SwI = dyn_cast<CatchSwitchInst>(I);
    Call = "CreateCatchSwitch(" + Op1 + ", " + Op2 + ", " +
           std::to_string(SwI->getNumHandlers()) + ")";

    std::string SwVar = asStr(SwI);
    OS << "auto " << SwVar << " = " << Builder << "." << Call << ";\n";
    for (auto C : SwI->handlers()) {
      OS << SwVar << "->addHandler(" << asStr(C) << ");\n";
    }
    return;
  } break;
  case Instruction::CallBr: {
    auto CallBRI = dyn_cast<CallBrInst>(I);
    std::string Inddest = "{";
    bool First = true;
    for (unsigned I = 0; I < CallBRI->getNumIndirectDests(); ++I) {
      if (!First)
        Inddest += ", ";
      Inddest += asStr(CallBRI->getIndirectDest(I));
      First = false;
    }
    Inddest += "}";

    std::string Args = "{";
    First = true;
    for (unsigned I = 0; I < CallBRI->arg_size(); ++I) {
      if (!First)
        Args += ", ";
      Args += asStr(CallBRI->getArgOperand(I));
      First = false;
    }
    Args += "}";
    Call = "CreateCallBr(" + asStr(CallBRI->getFunctionType()) + ", " +
           asStr(I->getOperand(I->getNumOperands() - 1)) + ", " +
           asStr(CallBRI->getDefaultDest()) + ", " + Inddest + ", " + Args +
           ")";
    // TODO: Handle operand bundles.
  } break;
  case Instruction::FNeg: {
    Call = "CreateFNeg(" + Op1 + ")";
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
    Call = "CreateUDiv(" + Op1 + ", " + Op2 + ", \"\", " +
           toStr(I->isExact()) + ")";
  } break;
  case Instruction::SDiv: {
    Call = "CreateSDiv(" + Op1 + ", " + Op2 + ", \"\", " +
           toStr(I->isExact()) + ")";
  } break;
  case Instruction::FDiv: {
    Call = getFPBinArithOp("FDiv", Op1, Op2, I);
  } break;
  case Instruction::URem: {
    Call = "CreateURem(" + Op1 + ", " + Op2 + ")";
  } break;
  case Instruction::SRem: {
    Call = "CreateSRem(" + Op1 + ", " + Op2 + ")";
  } break;
  case Instruction::FRem: {
    Call = getFPBinArithOp("FRem", Op1, Op2, I);
  } break;
  case Instruction::Shl: {
    Call = getBinArithOp("Shl", Op1, Op2, I);
  } break;
  case Instruction::LShr: {
    Call = "CreateLShr(" + Op1 + ", " + Op2 + ", \"\", " +
           toStr(I->isExact()) + ")";
  } break;
  case Instruction::AShr: {
    Call = "CreateAShr(" + Op1 + ", " + Op2 + ", \"\", " +
           toStr(I->isExact()) + ")";
  } break;
  case Instruction::And: {
    Call = "CreateAnd(" + Op1 + ", " + Op2 + ")";
  } break;
  case Instruction::Or: {
    Call = "CreateOr(" + Op1 + ", " + Op2 + ")";
  } break;
  case Instruction::Xor: {
    Call = "CreateXor(" + Op1 + ", " + Op2 + ")";
  } break;
  case Instruction::Alloca: {
    auto AlI = dyn_cast<AllocaInst>(I);
    auto Val = AlI->getArraySize();
    auto ValStr = Val ? asStr(Val) : "nullptr";
    Call = "CreateAlloca(" + asStr(AlI->getAllocatedType()) + ", " +
           std::to_string(AlI->getAddressSpace()) + ", " + ValStr + ")";
  } break;
  case Instruction::Load: {
    auto LI = dyn_cast<LoadInst>(I);
    Call = "CreateLoad(" + asStr(I->getType()) + ", " + Op1 + ", " +
           toStr(LI->isVolatile()) + ")";
  } break;
  case Instruction::Store: {
    auto SI = dyn_cast<StoreInst>(I);
    Call = "CreateStore(" + Op1 + ", " + Op2 + ", " + toStr(SI->isVolatile()) +
           ")";
  } break;
  case Instruction::GetElementPtr: {
    auto GEPI = dyn_cast<GetElementPtrInst>(I);
    std::string StrList = LLVMPrefix + "ArrayRef<" + LLVMPrefix + "Value*>(";
    if (I->getNumOperands() > 2)
      StrList += "{";
    bool First = true;
    for (unsigned J = 1; J < I->getNumOperands(); ++J) {
      if (!First)
        StrList += ", ";
      StrList += asStr(I->getOperand(J));
      First = false;
    }
    if (I->getNumOperands() > 2)
      StrList += "}";
    StrList += ")";
    Call = "CreateGEP(" + asStr(GEPI->getSourceElementType()) + ", " + Op1 +
           ", " + StrList + ", \"\", " + toStr(GEPI->isInBounds()) + ")";
  } break;
  case Instruction::Fence: {
    auto FI = dyn_cast<FenceInst>(I);
    Call = "CreateFence(" + asStr(FI->getOrdering()) + ", " +
           asStr(FI->getSyncScopeID()) + ")";
  } break;
  case Instruction::AtomicCmpXchg: {
    auto AcmpxI = dyn_cast<AtomicCmpXchgInst>(I);
    Call = "CreateAtomicCmpXchg(" + Op1 + ", " + Op2 + ", " + Op3 + ", Align(" +
           std::to_string(AcmpxI->getAlign().value()) + "), " +
           asStr(AcmpxI->getSuccessOrdering()) + ", " +
           asStr(AcmpxI->getFailureOrdering()) + ", " +
           asStr(AcmpxI->getSyncScopeID()) + ")";
  } break;
  case Instruction::AtomicRMW: {
    auto ArmwI = dyn_cast<AtomicRMWInst>(I);
    Call = "CreateAtomicRMW(" + asStr(ArmwI->getOperation()) + ", " +
           Op1 + ", " + Op2 + ", Align(" +
           std::to_string(ArmwI->getAlign().value()) + "), " +
           asStr(ArmwI->getOrdering()) + ", " +
           asStr(ArmwI->getSyncScopeID()) + ")";
  } break;
  case Instruction::Trunc: {
    Call = "CreateTrunc(" + Op1 + ", " + asStr(I->getType()) + ")";
  } break;
  case Instruction::ZExt: {
    Call = "CreateZExt(" + Op1 + ", " + asStr(I->getType()) + ", \"\", " +
           toStr(I->hasNonNeg()) + ")";
  } break;
  case Instruction::SExt: {
    Call = "CreateSExt(" + Op1 + ", " + asStr(I->getType()) + ")";
  } break;
  case Instruction::FPToUI: {
    Call = "CreateFPToUI(" + Op1 + ", " + asStr(I->getType()) + ")";
  } break;
  case Instruction::FPToSI: {
    Call = "CreateFPToSI(" + Op1 + ", " + asStr(I->getType()) + ")";
  } break;
  case Instruction::UIToFP: {
    Call = "CreateUIToFP(" + Op1 + ", " + asStr(I->getType()) + ")";
  } break;
  case Instruction::SIToFP: {
    Call = "CreateSIToFP(" + Op1 + ", " + asStr(I->getType()) + ")";
  } break;
  case Instruction::FPTrunc: {
    Call = "CreateFPTrunc(" + Op1 + ", " + asStr(I->getType()) + ")";
  } break;
  case Instruction::FPExt: {
    Call = "CreateFPExt(" + Op1 + ", " + asStr(I->getType()) + ")";
  } break;
  case Instruction::PtrToInt: {
    Call = "CreatePtrToInt(" + Op1 + ", " + asStr(I->getType()) + ")";
  } break;
  case Instruction::IntToPtr: {
    Call = "CreateIntToPtr(" + Op1 + ", " + asStr(I->getType()) + ")";
  } break;
  case Instruction::BitCast: {
    Call = "CreateBitCast(" + Op1 + ", " + asStr(I->getType()) + ")";
  } break;
  case Instruction::AddrSpaceCast: {
    Call = "CreateAddrSpaceCast(" + Op1 + ", " + asStr(I->getType()) + ")";
  } break;
  case Instruction::CleanupPad: {
    auto CupI = dyn_cast<CleanupPadInst>(I);
    std::string ArgsStr = "{";
    bool First = true;
    for (unsigned I = 0; I < CupI->arg_size(); ++I) {
      if (!First)
        ArgsStr += ", ";
      ArgsStr += asStr(CupI->getArgOperand(I));
      First = false;
    }
    ArgsStr += "}";
    Call = "CreateCleanupPad(" + Op1 + ", " + ArgsStr + ")";
  } break;
  case Instruction::CatchPad: {
    auto CapI = dyn_cast<CatchPadInst>(I);
    std::string ArgsStr = "{";
    bool First = true;
    for (unsigned I = 0; I < CapI->arg_size(); ++I) {
      if (!First)
        ArgsStr += ", ";
      ArgsStr += asStr(CapI->getArgOperand(I));
      First = false;
    }
    ArgsStr += "}";
    Call = "CreateCatchPad(" + Op1 + ", " + ArgsStr + ")";
  } break;
  case Instruction::ICmp: {
    auto CmpI = dyn_cast<CmpInst>(I);
    std::string CmpPredicate =
        LLVMPrefix + asStr(CmpI->getPredicate());
    Call = "CreateICmp(" + CmpPredicate + ", " + Op1 + ", " + Op2 + ")";
  } break;
  case Instruction::FCmp: {
    auto CmpI = dyn_cast<CmpInst>(I);
    std::string CmpPredicate =
        LLVMPrefix + asStr(CmpI->getPredicate());
    Call = "CreateFCmp(" + CmpPredicate + ", " + Op1 + ", " + Op2 + ")";
  } break;
  case Instruction::PHI: {
    auto PhI = dyn_cast<PHINode>(I);
    Call = "CreatePHI(" + asStr(I->getType()) + ", " +
           std::to_string(PhI->getNumIncomingValues()) + ")";
    std::string PhVar = asStr(PhI);
    OS << "auto " << PhVar << " = " << Builder << "." << Call << ";\n";

    unsigned I = 0;
    for (auto B : PhI->blocks()) {
      auto IncVal = asStr(PhI->getIncomingValue(I));
      // Phis might contain incoming not yet defined variable in such case
      // we save the line we would output here and output it after the
      // whole function body was outputted.
      std::string Line =
          PhVar + "->addIncoming(" + IncVal + ", " + asStr(B) + ");\n";
      PhiIncomings.push_back(Line);
      ++I;
    }
    return;
  } break;
  case Instruction::Call: {
    auto FcI = dyn_cast<CallBase>(I);
    std::string ArgDecl = "";
    if (FcI->arg_size() > 0) {
      ArgDecl = getNextVar();
      OS << "Value *" << ArgDecl << "[] = {";
      bool IsFirst = true;
      for (unsigned I = 0; I < FcI->arg_size(); ++I) {
        if (!IsFirst)
          OS << ", ";
        OS << asStr(FcI->getArgOperand(I));
        IsFirst = false;
      }
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

    auto Fun = dyn_cast<Function>(I->getOperand(I->getNumOperands() - 1));
    std::string FunDecl = getNextVar();
    if (!Fun)
      Fun = FcI->getCalledFunction();

    if (!Fun) {
      if (auto InA = dyn_cast<InlineAsm>(FcI->getCalledOperand())) {
        if (ArgDecl.empty()) {
          Call = "CreateCall(" + asStr(InA->getFunctionType()) + ", " +
               asStr(I->getOperand(I->getNumOperands() - 1)) + ")";
        } else {
          Call = "CreateCall(" + asStr(InA->getFunctionType()) + ", " +
                asStr(I->getOperand(I->getNumOperands() - 1)) + ", " + ArgDecl +
                ")";
        }
      }
      else if (FcI->isIndirectCall()) {
        if (ArgDecl.empty()) {
          Call = "CreateCall(" + asStr(FcI->getFunctionType()) + ", " +
                asStr(I->getOperand(I->getNumOperands() - 1)) + ")";
        } else {
          Call = "CreateCall(" + asStr(FcI->getFunctionType()) + ", " +
                asStr(I->getOperand(I->getNumOperands() - 1)) + ", " + ArgDecl +
                ")";
        }
      }
      else if (auto GA = dyn_cast<GlobalAlias>(FcI->getCalledOperand())) {
        Fun = dyn_cast<Function>(GA->getAliaseeObject());
        assert(Fun && "Global alias is not a function");
        OS << "auto " << FunDecl << " = " << ModName << "->getOrInsertFunction(\""
            << Fun->getName() << "\", " << asStr(Fun->getFunctionType())
            << ");\n";
        if (!ArgDecl.empty())
          Call = "CreateCall(" + FunDecl + ", " + ArgDecl + ")";
        else
          Call = "CreateCall(" + FunDecl + ")";
      }
      else {
        OS << "/* TODO: Unknown CallBase type in Call instruction */\n";
        return;
      }
    } else {
      // No need to save this variable as it is a temporary one.
      OS << "auto " << FunDecl << " = " << ModName << "->getOrInsertFunction(\""
         << Fun->getName() << "\", " << asStr(Fun->getFunctionType())
         << ");\n";
      if (!ArgDecl.empty())
        Call = "CreateCall(" + FunDecl + ", " + ArgDecl + ")";
      else
        Call = "CreateCall(" + FunDecl + ")";
    }
  } break;
  case Instruction::Select: {
    Call = "CreateSelect(" + Op1 + ", " + Op2 + ", " + Op3 + ")";
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
    Call = "CreateVAArg(" + Op1 + ", " + asStr(I->getType()) + ")";
  } break;
  case Instruction::ExtractElement: {
    Call = "CreateExtractElement(" + Op1 + ", " + Op2 + ")";
  } break;
  case Instruction::InsertElement: {
    Call = "CreateInsertElement(" + Op1 + ", " + Op2 + ", " + Op3 + ")";
  } break;
  case Instruction::ShuffleVector: {
    auto SvI = dyn_cast<ShuffleVectorInst>(I);
    std::string MaskStr = "{";
    bool First = true;
    for (int I : SvI->getShuffleMask()) {
      if (!First)
        MaskStr += ", ";
      MaskStr += std::to_string(I);
      First = false;
    }
    MaskStr += "}";
    Call = "CreateShuffleVector(" + Op1 + ", " + Op2 + ", " + MaskStr + ")";
  } break;
  case Instruction::ExtractValue: {
    auto EvI = dyn_cast<ExtractValueInst>(I);
    std::string ArgStr = "{";
    bool IsFirst = true;
    for (auto ind : EvI->getIndices()) {
      if (!IsFirst)
        ArgStr += ", ";
      ArgStr += std::to_string(ind);
      IsFirst = false;
    }
    ArgStr += "}";
    Call = "CreateExtractValue(" + Op1 + ", " + ArgStr + ")";
  } break;
  case Instruction::InsertValue: {
    auto IvI = dyn_cast<InsertValueInst>(I);
    std::string ArgStr = "{";
    bool IsFirst = true;
    for (auto Ind : IvI->getIndices()) {
      if (!IsFirst)
        ArgStr += ", ";
      ArgStr += std::to_string(Ind);
      IsFirst = false;
    }
    ArgStr += "}";
    Call = "CreateInsertValue(" + Op1 + ", " + Op2 + ", " + ArgStr + ")";
  } break;
  case Instruction::LandingPad: {
    auto LPI = dyn_cast<LandingPadInst>(I);
    std::string LPVar = asStr(I);
    Call = "CreateLandingPad(" + asStr(I->getType()) + ", " +
           std::to_string(LPI->getNumClauses()) + ")";
    OS << "auto " << LPVar << " = " << Builder << "." << Call << ";\n";
    for (unsigned I = 0; I < LPI->getNumClauses(); ++I) {
      OS << LPVar << "->addClause(" << asStr(LPI->getClause(I)) << ");\n";
    }
    return;
  } break;
  case Instruction::Freeze: {
    Call = "CreateFreeze(" + Op1 + ")";
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
