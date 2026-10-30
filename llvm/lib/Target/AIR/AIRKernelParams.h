//===- AIRKernelParams.h - Kernel values passed down to callees -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// AIR has values that only a kernel has: its system values (the thread's
/// position, its SIMD lane, ...) and the buffers the AIR target binds for it
/// (the lock table of emulated atomics, ...). They are kernel parameters. A
/// function a kernel calls (one AIRInlineNonKernel kept as a function) gets
/// them as parameters of its own, which its callers pass, up to the kernels.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRKERNELPARAMS_H
#define LLVM_LIB_TARGET_AIR_AIRKERNELPARAMS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace llvm {

class Argument;
class Function;
class Module;
class Type;

/// A value that function F needs from its kernel. Key names it (the
/// parameters of one key are the same value in every function); Name is the
/// name of the parameter that holds it.
struct AIRKernelParamRequest {
  Function *F;
  StringRef Key;
  Type *Ty;
  StringRef Name;
};

/// The outcome of addKernelParams.
struct AIRKernelParams {
  /// The functions that were replaced (to add parameters), old to new. The
  /// old ones have been erased.
  DenseMap<Function *, Function *> Replaced;
  /// For each function that has a requested value (the requesting functions
  /// and their callers, transitively), the argument holding each key.
  DenseMap<Function *, StringMap<Argument *>> Args;
  /// The parameters added to functions nothing calls (the kernels), for the
  /// caller to describe in the kernel's metadata: (kernel, key, parameter).
  SmallVector<std::tuple<Function *, std::string, Argument *>, 4> Added;

  /// The function F now is (F itself when it was not replaced).
  Function *get(Function *F) const { return Replaced.lookup_or(F, F); }
};

/// Gives each requesting function, and each of its callers transitively, a
/// parameter for each key it needs, and passes it at every call. A function
/// that already has the value (Existing returns its argument, e.g. a kernel
/// with a pre-baked system-value parameter) is not given another. New
/// parameters are appended, in the order of the requests. A function whose
/// address is used other than to call it cannot get one: that is a fatal
/// error.
AIRKernelParams addKernelParams(
    Module &M, ArrayRef<AIRKernelParamRequest> Requests,
    function_ref<Argument *(Function &, StringRef Key)> Existing = nullptr);

/// The name of kernel parameter A in the kernel's interface (air.arg_name):
/// its "air-arg-name" attribute, which survives where the context discards
/// value names (as release builds do) and module clones, or else its name.
StringRef airArgName(const Argument &A);

/// Names A, or, where the context discards value names, gives it the
/// "air-arg-name" attribute that airArgName reads.
void setAIRArgName(Argument &A, StringRef Name);

/// Whether F is called (directly) from a function, i.e. not a kernel.
bool isCalledFunction(const Function &F);

} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRKERNELPARAMS_H
