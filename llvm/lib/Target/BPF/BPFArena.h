//===-- BPFArena.h - What the arena passes have in common -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The passes that keep the memory of a program in arena recognize the kernel by
// what the IR says, not by names:
//
//   program   a function that has a section
//   helper    a call of a constant number
//   kfunc     a call of a declaration in section .ksyms
//   arena     'ptr addrspace(1)' in the type of a helper or a kfunc. All other
//             pointers that they take and return are pointers of the kernel.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_BPF_BPFARENA_H
#define LLVM_LIB_TARGET_BPF_BPFARENA_H

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class Argument;
class CallBase;
class Function;
class GlobalVariable;
class Module;
class Use;
class Value;

/// The address space of arena.
inline constexpr unsigned BPFArenaAS = 1;

/// Metadata of an alloca that is in the stack in arena.
inline constexpr StringLiteral BPFArenaStackMD = "bpf.arena.stack";
/// Metadata of an alloca that must not go to the stack in arena.
inline constexpr StringLiteral BPFStackKeepMD = "bpf.stack.keep";
/// Metadata of a load that yields a pointer to the stack of the program when it
/// reads the stack of the program.
inline constexpr StringLiteral BPFStackPtrMD = "bpf.stack.ptr";

/// The variable is in arena.
bool isBPFArenaGlobal(const GlobalVariable &GV);

/// The global variables in which the program stores a pointer of the kernel,
/// and does not use what it loads from there as a pointer. \p Moves says
/// which global variables are to move to arena. See BPFArenaKptr.cpp.
void findBPFKernelPtrGlobals(Module &M,
                             function_ref<bool(const GlobalVariable &)> Moves,
                             SmallPtrSetImpl<const GlobalVariable *> &GVs);

/// The call of a helper or of a kfunc.
bool isBPFKernelFunc(const CallBase &CB);

/// The argument of \p F for parameter number \p N of the source, from 0, or
/// nullptr. That is how debug info counts. A function that returns a struct in
/// memory has one more argument in front of them.
const Argument *getBPFSourceParam(const Function &F, unsigned N);

/// Helpers and kfuncs take memory that the kernel knows about: the stack of
/// the program, a map value. This finds the addresses that get to them, also
/// through the functions of the module, as a number, through memory, and in
/// memory that they get: the arguments of bpf_snprintf(), union bpf_attr,
/// the context of a callback. Inline asm is the same: it gets no cast, so it
/// cannot access arena through the address.
///
/// A number can be an address that the kernel reads ("%s") or data ("%lx"), and
/// the compiler cannot tell. So an address that gets there as a number or
/// through memory is weak evidence. It does not count if the address also goes
/// where only an address in arena makes sense: to the heap, to a global
/// variable, to another file. It does not change what a function that is not
/// static expects either.
class BPFKernelMem {
  /// What was looked at: as an address, and as the address of memory that
  /// holds one.
  struct Visited {
    SmallPtrSet<const Value *, 16> Addrs, Holders;
    enum {
      Direct,   ///< Follow calls and constant offsets.
      Pointers, ///< Also other offsets, select and phi.
      Weak,     ///< Also numbers and memory.
    } Mode = Weak;
  };

  /// Parameters that get to a helper, a kfunc or inline asm, and parameters
  /// that the source describes with a decl tag other than 'arg:arena'.
  SmallPtrSet<const Argument *, 16> Params;
  /// Parameters of static functions for which there is weak evidence only.
  SmallPtrSet<const Argument *, 16> WeakParams;

  bool isKernelUse(const Use &U, Visited &Seen) const;
  bool isPassed(const Value *Ptr, Visited &Seen) const;
  bool isContentPassed(const Value *Ptr, Visited &Seen) const;
  bool goesToArena(const Value *Ptr, Visited &Seen) const;
  bool isInArena(const Value *Obj, Visited &Seen) const;

public:
  explicit BPFKernelMem(const Module &M);

  /// The parameter points to memory that the kernel knows about.
  bool isKernelParam(const Argument &A) const { return Params.contains(&A); }
  /// The address in \p U, maybe with an offset, gets to one.
  bool isKernelUse(const Use &U) const;
  /// One of the uses of the address \p Ptr does, and the evidence counts.
  bool isPassed(const Value *Ptr) const;
};

} // namespace llvm

#endif
