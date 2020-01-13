//===- TapirTargetIDs.h - Tapir target ID's --------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file enumerates the available Tapir lowering targets.
//
//===----------------------------------------------------------------------===//

#ifndef TAPIR_TARGET_IDS_H_
#define TAPIR_TARGET_IDS_H_

namespace llvm {

enum class TapirTargetID {
  None,    // Perform no lowering
  Serial,  // FIXME: Make this option useful.  Perhaps only outline tasks?
  Cilk,    // Lower to the Cilk Plus ABI
  OpenMP,  // Lower to OpenMP
  CilkR,   // Lower to the CilkR ABI
  Cheetah, // Lower to the Cheetah ABI
  Last_TapirTargetID
};

} // end namespace llvm

#endif
