//===- BuildBuiltins.h - Utility builder for builtins ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements some functions for lowering compiler builtins,
// specifically for atomics. Currently, LLVM-IR has no representation of atomics
// that can be used independent of its arguments:
//
// * The instructions load atomic, store atomic, atomicrmw, and cmpxchg can only
//   be used with constant memory model, sync scope, data sizes (that must be
//   power-of-2), volatile and weak property, and should not be used with data
//   types that are untypically large which may slow down the compiler.
//
// * libcall (in GCC's case: libatomic; LLVM: Compiler-RT) functions work with
//   any data size, but are slower. Specialized functions for a selected number
//   of data sizes exist as well. They do not support sync scopes, the volatile
//   or weakness property. These functions may be implemented using a lock and
//   availability depends on the target triple (e.g. GPU devices cannot
//   implement a global lock by design).
//
// We want to mimic Clang's behaviour:
//
// * Prefer atomic instructions over libcall functions whenever possible. When a
//   target backend does not support atomic instructions natively,
//   AtomicExpandPass, LowerAtomicPass, or some backend-specific pass lower will
//   convert such instructions to a libcall function call. The reverse is not
//   the case, i.e. once a libcall function is emitted, there is no pass that
//   optimizes it into an instruction.
//
// * When passed a non-constant enum argument which the instruction requires to
//   be constant, then emit a switch case for each enum case.
//
// Clang currently doesn't actually check whether the target actually supports
// atomic libcall functions so it will always fall back to a libcall function
// even if the target does not support it. That is, emitting an atomic builtin
// may fail and a frontend needs to handle this case.
//
// Clang also assumes that the maximum supported data size of atomic instruction
// is 16, despite this is target-dependent and should be queried using
// TargetLowing::getMaxAtomicSizeInBitsSupported(). However, TargetMachine
// (which is a factory for TargetLowering) is not available during Clang's
// CodeGen phase, it is only created for the LLVM pass pipeline.
//
// The functions in this file are intended to handle the complexity of builtins
// so frontends do not need to care about the details. A major difference
// between the cases is that the IR instructions take values directly as an
// llvm::Value (except the atomic address of course), but the libcall functions
// almost always take pointers to those values. Since we cannot assume that
// everything can be passed an llvm::Value (LLVM does not handle large types
// such as i4096 well), our abstraction passes everything as pointer which is
// loaded when needed. The caller is responsible to emit a temporary AllocaInst
// and store if it needs to pass an llvm::Value. Mem2Reg/SROA will easily remove
// any unnecessary store/load pairs.
//
// In the future LLVM may introduce more generic atomic constructs that is
// lowered by an LLVM pass, such as AtomicExpandPass. Once this exist, the
// emitBuiltin functions in this file become trivial.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_BUILDBUILTINS_H
#define LLVM_TRANSFORMS_UTILS_BUILDBUILTINS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <variant>

namespace llvm {
class Value;
class TargetLibraryInfo;
class DataLayout;
class IRBuilderBase;
class Type;
class TargetLowering;

namespace SyncScope {
typedef uint8_t ID;
}

/// Options for controlling atomic builtins.
struct AtomicEmitOptions {
  AtomicEmitOptions(const DataLayout &DL, const TargetLibraryInfo *TLI,
                    const TargetLowering *TL = nullptr)
      : DL(DL), TLI(TLI), TL(TL) {}

  /// The target's data layout.
  const DataLayout &DL;

  /// The target's libcall library availability.
  const TargetLibraryInfo *TLI;

  /// Used to determine which instructions thetarget support. If omitted,
  /// assumes all accesses up to a size of 16 bytes are supported.
  const TargetLowering *TL = nullptr;

  /// Whether an LLVM instruction can be emitted. LLVM instructions include:
  ///  * load atomic
  ///  * store atomic
  ///  * cmpxchg
  ///  * atomicrmw
  ///
  /// Atomic LLVM intructions have several restrictions on when they can be
  /// used, including:
  ///  * Properties such as IsVolatile,IsWeak,Memorder,Scope must be constant.
  ///  * Must be an integer or pointer type. Some cases also allow float types.
  ///  * Size must be a power-of-two number of bytes.
  ///  * Size must be at most the size of atomics supported by the target.
  ///  * Size should not be too large (e.g. i4096) since LLVM does not scale
  ///    well with huge types.
  ///
  /// Even with all these limitations adhered to, AtomicExpandPass may still
  /// lower the instruction to a libcall function if the target does not support
  /// it.
  ///
  /// See also:
  ///  * https://llvm.org/docs/Atomics.html
  ///  * https://llvm.org/docs/LangRef.html#i-load
  ///  * https://llvm.org/docs/LangRef.html#i-store
  ///  * https://llvm.org/docs/LangRef.html#cmpxchg-instruction
  ///  * https://llvm.org/docs/LangRef.html#i-atomicrmw
  bool AllowInstruction = true;

  /// Whether a switch can be emitted to work around the requirement of
  /// properties of an instruction must be constant. That is, for each possible
  /// value of the property, jump to a version of that instruction encoding that
  /// property.
  bool AllowSwitch = true;

  /// Allow emitting calls to constant-sized libcall functions, such as
  ///  * __atomic_load_n
  ///  * __atomic_store_n
  ///  * __atomic_compare_exchange_n
  ///
  /// where n is as size supported by the target, typically 1,2,4,8,16
  ///
  /// See also:
  ///  * https://llvm.org/docs/Atomics.html
  ///  * https://gcc.gnu.org/wiki/Atomic/GCCMM/LIbrary#GCC_intrinsics
  bool AllowSizedLibcall = true;

  /// Allow emitting call to variable-sized libcall functions, such as
  // / * __atomic_load
  ///  * __atomic_store
  ///  * __atomic_compare_exchange
  ///
  /// Note that the signatures of these libcall functions are different from the
  /// compiler builtins of the same name.
  ///
  /// See also:
  ///  * https://llvm.org/docs/Atomics.html
  ///  * https://gcc.gnu.org/wiki/Atomic/GCCMM/LIbrary#GCC_intrinsics
  bool AllowLibcall = true;

  // TODO: Add additional lowerings:
  //  * __sync_* libcalls
  //  * Differently named atomic primitives
  //    (e.g. InterlockedCompareExchange, C11 primitives on Windows)
  //  * Using a lock implemention as last resort
};

/// Emit the __atomic_load builtin. This may either be lowered to the load LLVM
/// instruction, or to one of the following libcall functions: __atomic_load_1,
/// __atomic_load_2, __atomic_load_4, __atomic_load_8, __atomic_load_16,
/// __atomic_load.
///
/// Also see:
/// * https://llvm.org/docs/Atomics.html
/// * https://llvm.org/docs/LangRef.html#load-instruction
/// * https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html
/// * https://gcc.gnu.org/wiki/Atomic/GCCMM/LIbrary#GCC_intrinsics
///
/// @param AtomicPtr   The memory location accessed atomically.
/// @Param RetPtr      Pointer the the data to be loaded from \p Ptr.
/// @param TypeOrSize  Type of the value to be accessed. cmpxchg
///                    supports integer and pointers only, other atomics also
///                    support floats. If any other type or omitted, type-prunes
///                    to an integer the holds at least \p DataSize bytes.
///                    Alternatively, the number of bytes can be specified in
///                    which case an intergers is also used.
/// @param IsVolatile  Whether to mark the access as volatile.
/// @param Memorder    Memory model to be used for the affected atomic address.
/// @param Scope       (optional) The synchronization scope (domain of threads
///                    where this access has to be atomic, e.g. CUDA
///                    warp/block/grid-level atomics) of this access. Defaults
///                    to system scope.
/// @param Align       (optional) Known alignment of /p Ptr. If omitted,
///                    alignment is inferred from /p Ptr itself or falls back
///                    to no alignment.
/// @param Builder     Used to emit instructions.
/// @param EmitOptions For controlling what IR is emitted.
/// @param Name        (optional) Stem for generated instruction names.
///
/// @return An error if the atomic operation could not be emitted.
Error emitAtomicLoadBuiltin(
    Value *AtomicPtr, Value *RetPtr, std::variant<Type *, uint64_t> TypeOrSize,
    bool IsVolatile,
    std::variant<Value *, AtomicOrdering, AtomicOrderingCABI> Memorder,
    SyncScope::ID Scope, MaybeAlign Align, IRBuilderBase &Builder,
    AtomicEmitOptions EmitOptions, const Twine &Name = Twine());

/// Emit the __atomic_store builtin. It may either be lowered to the store LLVM
/// instruction, or to one of the following libcall functions: __atomic_store_1,
/// __atomic_store_2, __atomic_store_4, __atomic_store_8, __atomic_store_16,
/// __atomic_static.
///
/// Also see:
/// * https://llvm.org/docs/Atomics.html
/// * https://llvm.org/docs/LangRef.html#store-instruction
/// * https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html
/// * https://gcc.gnu.org/wiki/Atomic/GCCMM/LIbrary#GCC_intrinsics
///
/// @param AtomicPtr   The memory location accessed atomically.
/// @Param ValPtr      Pointer to the data to be stored at \p Ptr.
/// @param TypeOrSize  Type of the value to be accessed. cmpxchg
///                    supports integer and pointers only, other atomics also
///                    support floats. If any other type or omitted, type-prunes
///                    to an integer the holds at least \p DataSize bytes.
///                    Alternatively, the number of bytes can be specified in
///                    which case an intergers is also used.
/// @param IsVolatile  Whether to mark the access as volatile.
/// @param Memorder    Memory model to be used for the affected atomic address.
/// @param Scope       (optional) The synchronization scope (domain of threads
///                    where this access has to be atomic, e.g. CUDA
///                    warp/block/grid-level atomics) of this access. Defaults
///                    to system scope.
/// @param Align       (optional) Known alignment of /p Ptr. If omitted,
///                    alignment is inferred from /p Ptr itself or falls back
///                    to no alignment.
/// @param Builder     Used to emit instructions.
/// @param EmitOptions For controlling what IR is emitted.
/// @param Name        (optional) Stem for generated instruction names.
///
/// @return An error if the atomic operation could not be emitted.
Error emitAtomicStoreBuiltin(
    Value *AtomicPtr, Value *ValPtr, std::variant<Type *, uint64_t> TypeOrSize,
    bool IsVolatile,
    std::variant<Value *, AtomicOrdering, AtomicOrderingCABI> Memorder,
    SyncScope::ID Scope, MaybeAlign Align, IRBuilderBase &Builder,
    AtomicEmitOptions EmitOptions, const Twine &Name = Twine());

/// Emit the __atomic_compare_exchange builtin. This may either be
/// lowered to the cmpxchg LLVM instruction, or to one of the following libcall
/// functions: __atomic_compare_exchange_1, __atomic_compare_exchange_2,
/// __atomic_compare_exchange_4, __atomic_compare_exchange_8,
/// __atomic_compare_exchange_16, __atomic_compare_exchange.
///
/// Also see:
///  * https://llvm.org/docs/Atomics.html
///  * https://llvm.org/docs/LangRef.html#cmpxchg-instruction
///  * https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html
///  * https://gcc.gnu.org/wiki/Atomic/GCCMM/LIbrary#GCC_intrinsics
///
/// @param AtomicPtr   The memory location accessed atomically.
/// @Param ExpectedPtr Pointer to the data expected at \p Ptr. The exchange will
///                    only happen if the value at \p Ptr is equal to this
///                    (unless IsWeak is set). Data at \p ExpectedPtr may or may
///                    not be be overwritten, so do not use after this call.
/// @Param DesiredPtr  Pointer to the data that the data at \p Ptr is replaced
///                    with.
/// @param TypeOrSize  Type of the value to be accessed. cmpxchg
///                    supports integer and pointers only, other atomics also
///                    support floats. If any other type or omitted, type-prunes
///                    to an integer the holds at least \p DataSize bytes.
///                    Alternatively, the number of bytes can be specified in
///                    which case an intergers is also used.
/// @param IsWeak      If true, the exchange may not happen even if the data at
///                    \p Ptr equals to \p ExpectedPtr.
/// @param IsVolatile  Whether to mark the access as volatile.
/// @param SuccessMemorder If the exchange succeeds, memory is affected
///                    according to the memory model.
/// @param FailureMemorder If the exchange fails, memory is affected according
///                    to the memory model. It is considered an atomic "read"
///                    for the purpose of identifying release sequences. Must
///                    not be release, acquire-release, and at most as strong as
///                    \p SuccessMemorder.
/// @param Scope       (optional) The synchronization scope (domain of threads
///                    where this access has to be atomic, e.g. CUDA
///                    warp/block/grid-level atomics) of this access. Defaults
///                    to system scope.
/// @param ActualPtr   (optional) Receives the value at \p Ptr before the atomic
///                    exchange is attempted. This means:
///                    In case of success:
///                      The value at \p Ptr before the update. That is, the
///                      value passed behind \p ExpectedPtr.
///                    In case of failure
///                    (including spurious failures if IsWeak):
///                      The current value at \p Ptr, i.e. the operation
///                      effectively was an atomic load of that value using
///                      FailureMemorder semantics.
///                    Can be the same as ExpectedPtr in which case after the
///                    call returns \p ExpectedPtr/\p ActualPtr will be the
///                    value as defined above (in contrast to being undefined).
/// @param Align       (optional) Known alignment of /p Ptr. If omitted,
///                    alignment is inferred from /p Ptr itself or falls back
///                    to no alignment.
/// @param Builder     Used to emit instructions.
/// @param EmitOptions For controlling what IR is emitted.
/// @param Name        (optional) Stem for generated instruction names.
///
/// @return A boolean value that indicates whether the exchange has happened
///         (true) or not (false), or an error if the atomic operation could not
///         be emitted.
Expected<Value *> emitAtomicCompareExchangeBuiltin(
    Value *AtomicPtr, Value *ExpectedPtr, Value *DesiredPtr,
    std::variant<Type *, uint64_t> TypeOrSize,
    std::variant<Value *, bool> IsWeak, bool IsVolatile,
    std::variant<Value *, AtomicOrdering, AtomicOrderingCABI> SuccessMemorder,
    std::variant<std::monostate, Value *, AtomicOrdering, AtomicOrderingCABI>
        FailureMemorder,
    SyncScope::ID Scope, Value *PrevPtr, MaybeAlign Align,
    IRBuilderBase &Builder, AtomicEmitOptions EmitOptions,
    const Twine &Name = Twine());

} // namespace llvm

#endif /* LLVM_TRANSFORMS_UTILS_BUILDBUILTINS_H */
