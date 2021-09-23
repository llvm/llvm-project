//===-- target.h ---------- OpenMP device runtime target implementation ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Target region interfaces are simple interfaces designed to allow middle-end
// (=LLVM) passes to analyze and transform the code. To achieve good performance
// it may be required to run the associated passes. However, implementations of
// this interface shall always provide a correct implementation as close to the
// user expected code as possible.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPENMP_LIBOMPTARGET_DEVICERTLS_COMMON_TARGET_H
#define LLVM_OPENMP_LIBOMPTARGET_DEVICERTLS_COMMON_TARGET_H

#include <stdint.h>

extern "C" {

/// Forward declaration of the source location identifier "ident".
typedef struct ident ident_t;

/// The target region _kernel_ interface for GPUs
///
/// This deliberatly simple interface provides the middle-end (=LLVM) with
/// easier means to reason about the semantic of the code and transform it as
/// well. The runtime calls are therefore also desiged to carry sufficient
/// information necessary for optimizations.
///
///
/// Intended usage:
///
/// \code
/// void kernel(...) {
///   ThreadKind = __kmpc_target_init(Ident, /* Mode */ 1,
///                                   /* UseGenericStateMachine */ true,
///                                   /* RequiresFullRuntime */ ... );
///   if (ThreadKind == -1) {
///     // User defined kernel code.
///   }
///   __kmpc_target_deinit(...);
/// }
/// \endcode
///
/// Which can be transformed to:
///
/// \code
/// void kernel(...) {
///   ThreadKind = __kmpc_target_init(Ident, /* Mode */ 1,
///                                   /* UseGenericStateMachine */ false,
///                                   /* RequiresFullRuntime */ ... );
///   if (ThreadKind == -1) {
///     // User defined kernel code.
///   } else {
///     assume(ThreadKind == ThreadId);
///     // Custom, kernel-specific state machine code.
///   }
///   __kmpc_target_deinit(...);
/// }
/// \endcode
///
///
///{

/// Initialization
///
/// Must be called by all threads.
///
/// \param Ident               Source location identification, can be NULL.
///
int32_t __kmpc_target_init(ident_t *Ident, int8_t Mode,
                           bool UseGenericStateMachine,
                           bool RequiresFullRuntime);

/// De-Initialization
///
/// Must be called by the main thread in generic mode, can be called by all
/// threads. Must be called by all threads in SPMD mode.
///
/// In non-SPMD, this function releases the workers trapped in a state machine
/// and also any memory dynamically allocated by the runtime.
///
/// \param Ident Source location identification, can be NULL.
///
void __kmpc_target_deinit(ident_t *Ident, int8_t Mode,
                          bool RequiresFullRuntime);

///}
}
#endif
