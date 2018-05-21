(*===-- llvm_tapir_opts.ml - LLVM OCaml Interface -------------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

type tapir_target

(** Tapir pass to install Cilky stuff in place of detach/sync instructions. *)
external add_lower_tapir_to_cilk :
  [ `Module ] Llvm.PassManager.t -> tapir_target -> unit
  = "llvm_add_lower_tapir_to_cilk"

(** Tapir pass to spawn loops with recursive divide-and-conquer. *)
external add_loop_spawning :
  [ `Module ] Llvm.PassManager.t -> tapir_target -> unit
  = "llvm_add_loop_spawning"
