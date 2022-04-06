//===- DXILPrepare.cpp - Prepare LLVM Module for DXIL encoding ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains pases and utilities to convert a modern LLVM
/// module into a module compatible with the LLVM 3.7-based DirectX Intermediate
/// Language (DXIL).
//===----------------------------------------------------------------------===//

#include "DirectX.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Compiler.h"

#define DEBUG_TYPE "dxil-prepare"

using namespace llvm;

namespace {

constexpr bool isValidForDXIL(Attribute::AttrKind Attr) {
  return is_contained({Attribute::Alignment,
                       Attribute::AlwaysInline,
                       Attribute::Builtin,
                       Attribute::ByVal,
                       Attribute::InAlloca,
                       Attribute::Cold,
                       Attribute::Convergent,
                       Attribute::InlineHint,
                       Attribute::InReg,
                       Attribute::JumpTable,
                       Attribute::MinSize,
                       Attribute::Naked,
                       Attribute::Nest,
                       Attribute::NoAlias,
                       Attribute::NoBuiltin,
                       Attribute::NoCapture,
                       Attribute::NoDuplicate,
                       Attribute::NoImplicitFloat,
                       Attribute::NoInline,
                       Attribute::NonLazyBind,
                       Attribute::NonNull,
                       Attribute::Dereferenceable,
                       Attribute::DereferenceableOrNull,
                       Attribute::NoRedZone,
                       Attribute::NoReturn,
                       Attribute::NoUnwind,
                       Attribute::OptimizeForSize,
                       Attribute::OptimizeNone,
                       Attribute::ReadNone,
                       Attribute::ReadOnly,
                       Attribute::ArgMemOnly,
                       Attribute::Returned,
                       Attribute::ReturnsTwice,
                       Attribute::SExt,
                       Attribute::StackAlignment,
                       Attribute::StackProtect,
                       Attribute::StackProtectReq,
                       Attribute::StackProtectStrong,
                       Attribute::SafeStack,
                       Attribute::StructRet,
                       Attribute::SanitizeAddress,
                       Attribute::SanitizeThread,
                       Attribute::SanitizeMemory,
                       Attribute::UWTable,
                       Attribute::ZExt},
                      Attr);
}

class DXILPrepareModule : public ModulePass {
public:
  bool runOnModule(Module &M) override {
    AttributeMask AttrMask;
    for (Attribute::AttrKind I = Attribute::None; I != Attribute::EndAttrKinds;
         I = Attribute::AttrKind(I + 1)) {
      if (!isValidForDXIL(I))
        AttrMask.addAttribute(I);
    }
    for (auto &F : M.functions()) {
      F.removeFnAttrs(AttrMask);
      F.removeRetAttrs(AttrMask);
      for (size_t Idx = 0; Idx < F.arg_size(); ++Idx)
        F.removeParamAttrs(Idx, AttrMask);

      for (auto &BB : F) {
        IRBuilder<> Builder(&BB);
        for (auto &I : make_early_inc_range(BB)) {
          if (I.getOpcode() == Instruction::FNeg) {
            Builder.SetInsertPoint(&I);
            Value *In = I.getOperand(0);
            Value *Zero = ConstantFP::get(In->getType(), -0.0);
            I.replaceAllUsesWith(Builder.CreateFSub(Zero, In));
            I.eraseFromParent();
          }
        }
      }
    }
    return true;
  }

  DXILPrepareModule() : ModulePass(ID) {}

  static char ID; // Pass identification.
};
char DXILPrepareModule::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(DXILPrepareModule, DEBUG_TYPE, "DXIL Prepare Module",
                      false, false)
INITIALIZE_PASS_END(DXILPrepareModule, DEBUG_TYPE, "DXIL Prepare Module", false,
                    false)

ModulePass *llvm::createDXILPrepareModulePass() {
  return new DXILPrepareModule();
}
