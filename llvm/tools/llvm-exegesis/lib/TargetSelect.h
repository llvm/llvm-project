//===-- TargetSelect.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// Utilities to handle the creation of the native exegesis target.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_EXEGESIS_TARGET_SELECT_H
#define LLVM_TOOLS_LLVM_EXEGESIS_TARGET_SELECT_H

namespace llvm {
namespace exegesis {

#ifdef LLVM_EXEGESIS_INITIALIZE_NATIVE_TARGET
void LLVM_EXEGESIS_INITIALIZE_NATIVE_TARGET();
#endif

// Initializes the native exegesis target, or returns false if there is no
// native target (either because llvm-exegesis does not support the target or
// because it's not linked in).
inline bool InitializeNativeExegesisTarget() {
#ifdef LLVM_EXEGESIS_INITIALIZE_NATIVE_TARGET
  LLVM_EXEGESIS_INITIALIZE_NATIVE_TARGET();
  return true;
#else
  return false;
#endif
}

} // namespace exegesis
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_EXEGESIS_TARGET_SELECT_H
