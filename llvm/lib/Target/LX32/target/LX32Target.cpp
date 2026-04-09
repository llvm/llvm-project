//===-- LX32Target.cpp - LX32 Target Initialization -----------------------===//
//
// Part of the LX32 Project
// SPDX-License-Identifier: MIT
//
// This file is the single entry-point LLVM uses to initialize the target.
//
// Why this exists:
//   LLVM splits target initialization into multiple optional layers:
//   - TargetInfo  (Target triple/name registration)
//   - Target      (TargetMachine registration)
//   - TargetMC    (MC layer: asm info, instr info, inst printer, ...)
//
// If TargetMC isn't initialized, llc can still *recognize* -march=lx32 but
// may crash later when it tries to create an assembly streamer.
//
// Keeping this file tiny and explicit makes the backend robust and avoids
// "half-registered" targets.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Target-layer initialization entry point for LX32.
// It is organized into the following sections:
//
//   Section 0 — External includes and registration dependencies
//   Section 1 — LLVMInitializeLX32Target implementation
//
//===----------------------------------------------------------------------===//

#include "TargetInfo/LX32TargetInfo.h"

#include "../core/LX32TargetMachine.h"

#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"

// This file provides the *Target* layer entry point.
//
// Important:
//   Do NOT try to call the function from itself to "bundle" initialization.
//   LLVM drivers (llc/opt/clang) will call the individual init functions:
//     - LLVMInitializeLX32TargetInfo()
//     - LLVMInitializeLX32Target()
//     - LLVMInitializeLX32TargetMC()
//
// The MC layer initializer lives in mc/LX32MCTargetDesc.cpp.

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void LLVMInitializeLX32Target() {
  // Register the TargetMachine so llc/clang can construct it for -march=lx32.
  //
  // Keep this boring and explicit: all target policy/configuration belongs in
  // LX32TargetMachine; this file should only perform registration.
  RegisterTargetMachine<llvm::LX32TargetMachine> X(llvm::getTheLX32TargetInfo());
}

