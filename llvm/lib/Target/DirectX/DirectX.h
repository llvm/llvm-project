//===- DirectXTargetMachine.h - DirectX Target Implementation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DIRECTX_DIRECTX_H
#define LLVM_LIB_TARGET_DIRECTX_DIRECTX_H

namespace llvm {
class ModulePass;
class PassRegistry;

/// Initializer for dxil writer pass
void initializeWriteDXILPassPass(PassRegistry &);

/// Initializer for DXIL-prepare
void initializeDXILPrepareModulePass(PassRegistry &);

/// Pass to convert modules into DXIL-compatable modules
ModulePass *createDXILPrepareModulePass();

/// Initializer for DXILOpLowering
void initializeDXILOpLoweringLegacyPass(PassRegistry &);

/// Pass to lowering LLVM intrinsic call to DXIL op function call.
ModulePass *createDXILOpLoweringLegacyPass();

} // namespace llvm

#endif // LLVM_LIB_TARGET_DIRECTX_DIRECTX_H
