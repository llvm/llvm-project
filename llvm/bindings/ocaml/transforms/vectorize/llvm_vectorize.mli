(*===-- llvm_vectorize.mli - LLVM OCaml Interface -------------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

(** Vectorize Transforms.

    This interface provides an OCaml API for LLVM vectorize transforms, the
    classes in the [LLVMVectorize] library. *)

(** See the [llvm::createLoopVectorizePass] function. *)
external add_loop_vectorize
  : [<Llvm.PassManager.any] Llvm.PassManager.t -> unit
  = "llvm_add_loop_vectorize"

(** See the [llvm::createSLPVectorizerPass] function. *)
external add_slp_vectorize
  : [<Llvm.PassManager.any] Llvm.PassManager.t -> unit
  = "llvm_add_slp_vectorize"
