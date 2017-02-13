//===-- AMDGPULowerIntrinsics.cpp -----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"

#define DEBUG_TYPE "amdgpu-lower-intrinsics"

using namespace llvm;

namespace {

const unsigned MaxStaticSize = 1024;

class AMDGPULowerIntrinsics : public ModulePass {
public:
  static char ID;

  AMDGPULowerIntrinsics() : ModulePass(ID) { }
  bool runOnModule(Module &M) override;
  StringRef getPassName() const override {
    return "AMDGPU Lower Intrinsics";
  }
};

}

char AMDGPULowerIntrinsics::ID = 0;

char &llvm::AMDGPULowerIntrinsicsID = AMDGPULowerIntrinsics::ID;

INITIALIZE_PASS(AMDGPULowerIntrinsics, DEBUG_TYPE,
                "Lower intrinsics", false, false)

// TODO: Should refine based on estimated number of accesses (e.g. does it
// require splitting based on alignment)
static bool shouldExpandOperationWithSize(Value *Size) {
  ConstantInt *CI = dyn_cast<ConstantInt>(Size);
  return !CI || (CI->getZExtValue() > MaxStaticSize);
}

static bool expandMemIntrinsicUses(Function &F) {
  Intrinsic::ID ID = F.getIntrinsicID();
  bool Changed = false;

  for (auto I = F.user_begin(), E = F.user_end(); I != E;) {
    Instruction *Inst = cast<Instruction>(*I);
    ++I;

    switch (ID) {
    case Intrinsic::memcpy: {
      auto *Memcpy = cast<MemCpyInst>(Inst);
      if (shouldExpandOperationWithSize(Memcpy->getLength())) {
        expandMemCpyAsLoop(Memcpy);
        Changed = true;
        Memcpy->eraseFromParent();
      }

      break;
    }
    case Intrinsic::memmove: {
      auto *Memmove = cast<MemMoveInst>(Inst);
      if (shouldExpandOperationWithSize(Memmove->getLength())) {
        expandMemMoveAsLoop(Memmove);
        Changed = true;
        Memmove->eraseFromParent();
      }

      break;
    }
    case Intrinsic::memset: {
      auto *Memset = cast<MemSetInst>(Inst);
      if (shouldExpandOperationWithSize(Memset->getLength())) {
        expandMemSetAsLoop(Memset);
        Changed = true;
        Memset->eraseFromParent();
      }

      break;
    }
    default:
      break;
    }
  }

  return Changed;
}

bool AMDGPULowerIntrinsics::runOnModule(Module &M) {
  bool Changed = false;

  for (Function &F : M) {
    if (!F.isDeclaration())
      continue;

    switch (F.getIntrinsicID()) {
    case Intrinsic::memcpy:
    case Intrinsic::memmove:
    case Intrinsic::memset:
      if (expandMemIntrinsicUses(F))
        Changed = true;
      break;
    default:
      break;
    }
  }

  return Changed;
}

ModulePass *llvm::createAMDGPULowerIntrinsicsPass() {
  return new AMDGPULowerIntrinsics();
}
