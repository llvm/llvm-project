//===- TapirTypes.h - Tapir types ------------------------------*- C++ -*--===//
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

#ifndef TAPIR_TYPES_H_
#define TAPIR_TYPES_H_

namespace llvm {

enum class TapirTargetType {
  None,    // Perform no lowering
  Serial,  // FIXME: Make this option useful.  Perhaps only outline tasks?
  Cilk,    // Lower to the Cilk Plus ABI
  OpenMP,  // Lower to OpenMP
  CilkR,   // Lower to the CilkR ABI
  Last_TapirTargetType
};

} // end namespace llvm

#endif
