//===-- PISAVerifier.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PISA.h"
#include "PISATargetMachine.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsPISA.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PISAIntrinsicUtils.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/PISAAddrSpace.h"

#define DEBUG_TYPE "pisa-verifier"
#define DEBUG_NAME "PISA verifier"

using namespace llvm;

namespace {

class PISAVerifier : public ModulePass, public InstVisitor<PISAVerifier> {
public:
  static char ID;
  PISAVerifier() : ModulePass(ID) {}

  StringRef getPassName() const override { return DEBUG_NAME; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.setPreservesAll();
  }

  bool runOnModule(Module &M) override;

  void visitIntrinsicInst(IntrinsicInst &I);
  void visitStoreInst(StoreInst &I);
  void visitAtomicRMWInst(AtomicRMWInst &I);

private:
  void verifyFunction(Function &F);
  void verifyGlobalVariable(GlobalVariable &GV);
  void verifyKernelArg(Argument &Arg);
  void verifyRoundingMode(IntrinsicInst &I, bool HasSaturation = true);
  void verifyEnumArg(IntrinsicInst &I, unsigned ArgIdx, unsigned MaxVal,
                     StringRef ArgName);
  void verifyHostAccessMetadata(const GlobalVariable &GV,
                                ArrayRef<const MDNode *> MDs);

  void illegal(Twine Message) {
    assert(Ctx);
    Ctx->diagnose(DiagnosticInfoGeneric({Twine("PISA Verifier: ") + Message}));
  }

  void warning(Twine Message) {
    assert(Ctx);
    Ctx->diagnose(DiagnosticInfoGeneric(
        {Twine("PISA Verifier: ") + Message, DS_Warning}));
  }

  SmallSet<StringRef, 4> HostAccessNamesSeen;
  LLVMContext *Ctx = nullptr;
  const TargetMachine *TM = nullptr;
  const Function *CurrFunc = nullptr;
};

} // namespace

char PISAVerifier::ID = 0;
INITIALIZE_PASS_BEGIN(PISAVerifier, DEBUG_TYPE, DEBUG_NAME, false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(PISAVerifier, DEBUG_TYPE, DEBUG_NAME, false, false)

void PISAVerifier::verifyKernelArg(Argument &Arg) {
  // According to PISA spec:
  // - Kernel arguments are modeled as being flattened out and stored in
  // abstract location pointed to by corresponding kernel argument with actual
  // storage location of kernel parameters being implementation defined.
  // - Kernel parameters are read-only.
  // If a kernel argument is passed by value, PISA backend
  // substitutes the pointer argument with the actual value it points to.
  // Based on all the above:
  // - byref arguments have no sense in PISA and are not expected
  // - byval arguments are expected to be in the private addrspace in order to
  // allow creation of copies
  if (!Arg.getType()->isPointerTy())
    return;
  if (Arg.hasByRefAttr())
    illegal(Twine("Kernel pointer arguments with the byref attribute are not "
                  "allowed\nKernel: ") +
            Twine(Arg.getParent()->getName()) +
            "\nArg no: " + Twine(Arg.getArgNo()));
  else if (Arg.hasByValAttr() &&
           Arg.getType()->getPointerAddressSpace() !=
               static_cast<unsigned>(PISAAS::AddressSpace::PRIVATE))
    illegal(Twine("Kernel pointer arguments with byval attribute are "
                  "expected to be in the private addrspace\nKernel: ") +
            Twine(Arg.getParent()->getName()) +
            "\nArg no: " + Twine(Arg.getArgNo()));
}

void PISAVerifier::verifyRoundingMode(IntrinsicInst &I, bool HasSaturation) {
  const llvm::Function *F = I.getCalledFunction();
  auto RndOpndIdx =
      HasSaturation ? I.getNumOperands() - 3 : I.getNumOperands() - 2;
  auto RndValue = static_cast<llvm::RoundingMode>(
      cast<ConstantInt>(I.getOperand(RndOpndIdx))->getZExtValue());
  switch (RndValue) {
  case llvm::RoundingMode::TowardZero:
  case llvm::RoundingMode::NearestTiesToEven:
  case llvm::RoundingMode::TowardPositive:
  case llvm::RoundingMode::TowardNegative:
  case llvm::RoundingMode::NearestTiesToAway:
  case llvm::RoundingMode::Invalid:
    break;
  default:
    illegal("Intrinsic " + F->getName() +
            " specifies invalid rounding mode value " +
            std::to_string(static_cast<int>(RndValue)));
  }
}

void PISAVerifier::verifyEnumArg(IntrinsicInst &I, unsigned ArgIdx,
                                 unsigned MaxVal, StringRef ArgName) {
  const llvm::Function *F = I.getCalledFunction();
  auto Val = cast<ConstantInt>(I.getArgOperand(ArgIdx))->getZExtValue();
  if (Val > MaxVal)
    illegal("Intrinsic " + F->getName() + " has invalid " + ArgName +
            " value " + std::to_string(Val));
}

void PISAVerifier::visitIntrinsicInst(IntrinsicInst &I) {
  auto IID = I.getIntrinsicID();
  switch (IID) {
  case Intrinsic::log:
  case Intrinsic::log2:
  case Intrinsic::log10:
  case Intrinsic::exp:
  case Intrinsic::sin:
  case Intrinsic::cos:
  case Intrinsic::pow:
  case Intrinsic::powi:
    if (I.getType()->isDoubleTy()) {
      const llvm::Function *F = I.getCalledFunction();
      assert(F && "Intrinsic must have a called function");
      illegal("Intrinsic " + F->getName() +
              " is not supported on the PISA target");
    }
    break;
  case Intrinsic::pisa_bfn: {
    uint8_t Lut = static_cast<uint8_t>(
        cast<ConstantInt>(I.getArgOperand(0))->getZExtValue());
    if (Lut == 0x00)
      illegal("BFN operation 0x00 is invalid; valid range is 0x01 to 0xfe");
    else if (Lut == 0xff)
      illegal("BFN operation 0xff is invalid; valid range is 0x01 to 0xfe");
    break;
  }
  case Intrinsic::pisa_fadd:
  case Intrinsic::pisa_fsub:
  case Intrinsic::pisa_fmul:
  case Intrinsic::pisa_fma:
  case Intrinsic::pisa_sitofp:
  case Intrinsic::pisa_uitofp:
  case Intrinsic::pisa_ftrunc:
    verifyRoundingMode(I);
    break;
  case Intrinsic::pisa_fdiv_rnd:
  case Intrinsic::pisa_pow_rnd:
  case Intrinsic::pisa_fsqrt_rnd:
  case Intrinsic::pisa_frnd_rnd:
  case Intrinsic::pisa_frcp_rnd:
  case Intrinsic::pisa_sin_rnd:
  case Intrinsic::pisa_cos_rnd:
  case Intrinsic::pisa_tanh_rnd:
  case Intrinsic::pisa_exp_rnd:
  case Intrinsic::pisa_exp2_rnd:
  case Intrinsic::pisa_log_rnd:
  case Intrinsic::pisa_log2_rnd:
  case Intrinsic::pisa_log10_rnd:
  case Intrinsic::pisa_fptosi_rnd:
  case Intrinsic::pisa_fptoui_rnd:
    verifyRoundingMode(I, /*HasSaturation=*/false);
    break;
  case Intrinsic::pisa_shfl:
    verifyEnumArg(I, 0, pisa::SHFLMode::Last - 1, "shfl mode");
    break;
  case Intrinsic::pisa_ired:
    verifyEnumArg(I, 0, pisa::IRedOp::Last - 1, "ired op");
    break;
  case Intrinsic::pisa_fred:
    verifyEnumArg(I, 0, pisa::FRedOp::Last - 1, "fred op");
    break;
  default:
    break;
  }
}

void PISAVerifier::visitStoreInst(StoreInst &I) {
  if (I.getPointerAddressSpace() ==
      static_cast<unsigned>(PISAAS::AddressSpace::CONSTANT))
    illegal("Store to constant memory is not allowed");
}

void PISAVerifier::visitAtomicRMWInst(AtomicRMWInst &I) {
  if (I.getPointerAddressSpace() ==
      static_cast<unsigned>(PISAAS::AddressSpace::CONSTANT))
    illegal("AtomicRMW on constant memory is not allowed");
}

void PISAVerifier::verifyFunction(Function &F) {
  CurrFunc = &F;
  if (F.getCallingConv() == CallingConv::PISA_KERNEL)
    for (auto &Arg : F.args())
      verifyKernelArg(Arg);

  visit(F);
}

// The frontend attaches !intel_host_access metadata to a global variable to
// describe its host-side visibility. The metadata node has two operands: the
// host access mode (a 32-bit integer in the range [0, 3]) and the host-visible
// name (a string). The backend lowers it to the ".host_access" variable
// directive documented in the PISA spec (intel.github.io/pisa/variables.html).
void PISAVerifier::verifyHostAccessMetadata(const GlobalVariable &GV,
                                            ArrayRef<const MDNode *> MDs) {
  if (MDs.size() != 1) {
    illegal("!intel_host_access metadata attached more than once to global '" +
            GV.getName() + "'");
    return;
  }

  const MDNode *MD = MDs[0];
  if (MD->getNumOperands() != 2) {
    illegal("!intel_host_access metadata must have exactly 2 operands");
    return;
  }

  auto VerifyFirstOp = [&]() -> bool {
    auto *HostAccessVal = mdconst::dyn_extract<ConstantInt>(MD->getOperand(0));
    if (!HostAccessVal)
      return false;

    if (HostAccessVal->getBitWidth() != 32)
      return false;

    if (HostAccessVal->getZExtValue() > 3)
      return false;

    return true;
  };

  if (!VerifyFirstOp())
    illegal("Host access mode (first operand) of !intel_host_access metadata "
            "must be a 32-bit integer in the range [0, 3].");

  auto *NameMD = dyn_cast<MDString>(MD->getOperand(1));
  if (!NameMD) {
    illegal(
        "Host name (second operand) of !intel_host_access metadata must be a "
        "string.");
    return;
  }

  const bool Inserted = HostAccessNamesSeen.insert(NameMD->getString()).second;
  if (!Inserted)
    illegal("Host access name '" + NameMD->getString() +
            "' is specified for more than one global variable");
}

void PISAVerifier::verifyGlobalVariable(GlobalVariable &GV) {
  SmallVector<MDNode *, 1> MDs;
  const unsigned HostAccessKindId = Ctx->getMDKindID("intel_host_access");
  if (GV.getMetadata(HostAccessKindId, MDs); !MDs.empty())
    verifyHostAccessMetadata(GV, MDs);
}

bool PISAVerifier::runOnModule(Module &M) {
  Ctx = &M.getContext();
  auto &TPC = getAnalysis<TargetPassConfig>();
  TM = &TPC.getTM<TargetMachine>();
  for (Function &F : M)
    verifyFunction(F);
  for (GlobalVariable &GV : M.globals())
    verifyGlobalVariable(GV);
  return false;
}

ModulePass *llvm::createPISAVerifierPass() { return new PISAVerifier(); }
