(*===-- llvm_bitwriter.mli - LLVM Ocaml Interface ---------------*- C++ -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

(** Bitcode writer.

    This interface provides an ocaml API for the LLVM bitcode writer, the
    classes in the Bitwriter library. *)

(** [write_bitcode_file m path] writes the bitcode for module [m] to the file at
    [path]. Returns [true] if successful, [false] otherwise. *)
external write_bitcode_file : Llvm.llmodule -> string -> bool
                            = "llvm_write_bitcode_file"
