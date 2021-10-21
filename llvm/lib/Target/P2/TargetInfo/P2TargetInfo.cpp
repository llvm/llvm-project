//===-- P2TargetInfo.cpp - P2 Target Implementation -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TargetInfo/P2TargetInfo.h"
#include "llvm/MC/TargetRegistry.h"

namespace llvm {
    Target &getTheP2Target() {
        static Target TheP2Target;
        return TheP2Target;
    }
} // namespace llvm

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeP2TargetInfo() {
    llvm::RegisterTarget<llvm::Triple::p2> X(llvm::getTheP2Target(), "p2", "Propeller 2", "P2");
}