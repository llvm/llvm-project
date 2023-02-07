(*===-- llvm_ipo.mli - LLVM OCaml Interface -------------------*- OCaml -*-===*
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===----------------------------------------------------------------------===*)

(** IPO Transforms.

    This interface provides an OCaml API for LLVM interprocedural optimizations, the
    classes in the [LLVMIPO] library. *)

(** See the [llvm::createConstantMergePass] function. *)
external add_constant_merge
  : [ `Module ] Llvm.PassManager.t -> unit
  = "llvm_add_constant_merge"

(** See the [llvm::createDeadArgEliminationPass] function. *)
external add_dead_arg_elimination
  : [ `Module ] Llvm.PassManager.t -> unit
  = "llvm_add_dead_arg_elimination"

(** See the [llvm::createFunctionAttrsPass] function. *)
external add_function_attrs
  : [ `Module ] Llvm.PassManager.t -> unit
  = "llvm_add_function_attrs"

(** See the [llvm::createFunctionInliningPass] function. *)
external add_function_inlining
  : [ `Module ] Llvm.PassManager.t -> unit
  = "llvm_add_function_inlining"

(** See the [llvm::createAlwaysInlinerPass] function. *)
external add_always_inliner
  : [ `Module ] Llvm.PassManager.t -> unit
  = "llvm_add_always_inliner"

(** See the [llvm::createGlobalDCEPass] function. *)
external add_global_dce
  : [ `Module ] Llvm.PassManager.t -> unit
  = "llvm_add_global_dce"

(** See the [llvm::createGlobalOptimizerPass] function. *)
external add_global_optimizer
  : [ `Module ] Llvm.PassManager.t -> unit
  = "llvm_add_global_optimizer"

(** See the [llvm::createIPSCCPPass] function. *)
external add_ipsccp
  : [ `Module ] Llvm.PassManager.t -> unit
  = "llvm_add_ipsccp"

(** See the [llvm::createInternalizePass] function. *)
external add_internalize
  : [ `Module ] Llvm.PassManager.t -> all_but_main:bool -> unit
  = "llvm_add_internalize"

(** See the [llvm::createStripDeadPrototypesPass] function. *)
external add_strip_dead_prototypes
  : [ `Module ] Llvm.PassManager.t -> unit
  = "llvm_add_strip_dead_prototypes"

(** See the [llvm::createStripSymbolsPass] function. *)
external add_strip_symbols
  : [ `Module ] Llvm.PassManager.t -> unit
  = "llvm_add_strip_symbols"
