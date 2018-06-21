(*===-- llvm_vectorize.ml - LLVM OCaml Interface --------------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

external add_bb_vectorize
  : [<Llvm.PassManager.any] Llvm.PassManager.t -> unit
  = "llvm_add_bb_vectorize"
external add_loop_vectorize
  : [<Llvm.PassManager.any] Llvm.PassManager.t -> unit
  = "llvm_add_loop_vectorize"
external add_slp_vectorize
  : [<Llvm.PassManager.any] Llvm.PassManager.t -> unit
  = "llvm_add_slp_vectorize"
