(*===-- llvm_ipo.mli - LLVM Ocaml Interface ------------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

(** IPO Transforms.

    This interface provides an ocaml API for LLVM interprocedural optimizations, the
    classes in the [LLVMIPO] library. *)

(** See llvm::createAddArgumentPromotionPass *)
external add_argument_promotion : [ | `Module ] Llvm.PassManager.t -> unit =

  "llvm_add_argument_promotion"
(** See llvm::createConstantMergePass function. *)
external add_constant_merge : [ | `Module ] Llvm.PassManager.t -> unit =
  "llvm_add_constant_merge"

(**  See llvm::createDeadArgEliminationPass function. *)
external add_dead_arg_elimination :
  [ | `Module ] Llvm.PassManager.t -> unit = "llvm_add_dead_arg_elimination"

(**  See llvm::createFunctionAttrsPass function. *)
external add_function_attrs : [ | `Module ] Llvm.PassManager.t -> unit =
  "llvm_add_function_attrs"

(**  See llvm::createFunctionInliningPass function. *)
external add_function_inlining : [ | `Module ] Llvm.PassManager.t -> unit =
  "llvm_add_function_inlining"

(**  See llvm::createGlobalDCEPass function. *)
external add_global_dce : [ | `Module ] Llvm.PassManager.t -> unit =
  "llvm_add_global_dce"

(**  See llvm::createGlobalOptimizerPass function. *)
external add_global_optimizer : [ | `Module ] Llvm.PassManager.t -> unit =
  "llvm_add_global_optimizer"

(**  See llvm::createIPConstantPropagationPass function. *)
external add_ipc_propagation : [ | `Module ] Llvm.PassManager.t -> unit =
  "llvm_add_ipc_propagation"

(**  See llvm::createPruneEHPass function. *)
external add_prune_eh : [ | `Module ] Llvm.PassManager.t -> unit =
  "llvm_add_prune_eh"

(**  See llvm::createIPSCCPPass function. *)
external add_ipsccp : [ | `Module ] Llvm.PassManager.t -> unit =
  "llvm_add_ipsccp"

(**  See llvm::createInternalizePass function. *)
external add_internalize : [ | `Module ] Llvm.PassManager.t -> bool -> unit =
  "llvm_add_internalize"

(**  See llvm::createStripDeadPrototypesPass function. *)
external add_strip_dead_prototypes :
  [ | `Module ] Llvm.PassManager.t -> unit = "llvm_add_strip_dead_prototypes"

(**  See llvm::createStripSymbolsPass function. *)
external add_strip_symbols : [ | `Module ] Llvm.PassManager.t -> unit =
  "llvm_add_strip_symbols"
