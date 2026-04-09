//===-- LX32TargetInfo.h - LX32 Target Implementation -------------------===//
//
// Part of the LX32 Project
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Declares the target-info entry points for LX32.
//
// TargetInfo is the first layer in LLVM target initialization. It provides:
//   - the singleton Target object backing the lx32 backend
//   - registration function invoked by LLVMInitializeAllTargetInfos()
//
// It is organized into the following sections:
//
//   Section 0 — Forward declarations
//   Section 1 — TargetInfo entry-point declarations
//
#ifndef LLVM_LIB_TARGET_LX32_TARGETINFO_LX32TARGETINFO_H
#define LLVM_LIB_TARGET_LX32_TARGETINFO_LX32TARGETINFO_H

namespace llvm {
class Target;

Target &getTheLX32TargetInfo();

} // namespace llvm

#endif // LLVM_LIB_TARGET_LX32_TARGETINFO_LX32TARGETINFO_H