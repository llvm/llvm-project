(*===-- llvm_linker.mli - LLVM OCaml Interface -----------------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

(** Linker.

    This interface provides an OCaml API for LLVM bitcode linker,
    the classes in the Linker library. *)

exception Error of string

(** [link_modules dst src mode] links [src] into [dst], raising [Error]
    if the linking fails. *)
val link_modules : Llvm.llmodule -> Llvm.llmodule -> unit