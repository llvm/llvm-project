//===- TapirTargetIDs.h - Tapir target ID's --------------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
