(*===-- llvm_vectorize.mli - LLVM OCaml Interface -------------*- OCaml -*-===*
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
