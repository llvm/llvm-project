(*===-- llvm_scalar_opts.mli - LLVM Ocaml Interface ------------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

(** Scalar Transforms.

    This interface provides an ocaml API for LLVM scalar transforms, the
    classes in the [LLVMScalarOpts] library. *)

(** See the [llvm::createConstantPropogationPass] function. *)
external add_constant_propagation : [<Llvm.PassManager.any] Llvm.PassManager.t
                                    -> unit
                                  = "llvm_add_constant_propagation"

(** See the [llvm::createInstructionCombiningPass] function. *)
external add_instruction_combining : [<Llvm.PassManager.any] Llvm.PassManager.t
                                     -> unit
                                   = "llvm_add_instruction_combining"

(** See the [llvm::createPromoteMemoryToRegisterPass] function. *)
external
add_memory_to_register_promotion : [<Llvm.PassManager.any] Llvm.PassManager.t
                                   -> unit
                                 = "llvm_add_memory_to_register_promotion"

(** See the [llvm::createDemoteMemoryToRegisterPass] function. *)
external
add_memory_to_register_demotion : [<Llvm.PassManager.any] Llvm.PassManager.t
                                  -> unit
                                = "llvm_add_memory_to_register_demotion"

(** See the [llvm::createReassociatePass] function. *)
external add_reassociation : [<Llvm.PassManager.any] Llvm.PassManager.t
                             -> unit
                           = "llvm_add_reassociation"

(** See the [llvm::createGVNPass] function. *)
external add_gvn : [<Llvm.PassManager.any] Llvm.PassManager.t
                   -> unit
                 = "llvm_add_gvn"

(** See the [llvm::createCFGSimplificationPass] function. *)
external add_cfg_simplification : [<Llvm.PassManager.any] Llvm.PassManager.t
                                  -> unit
                                = "llvm_add_cfg_simplification"
