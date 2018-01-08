//===- TapirTypes.h - Tapir types               ----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is a simple pass wrapper around the PromoteMemToReg function call
// exposed by the Utils library.
//
//===----------------------------------------------------------------------===//

#ifndef TAPIR_TYPES_H_
#define TAPIR_TYPES_H_

namespace llvm {
class TapirTarget;
enum class TapirTargetType : int {
  None = 0,
  Serial = 1,
  Cilk = 2,
  OpenMP = 3,
  CilkR = 4,
};
}

#endif
