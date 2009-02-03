(*===-- llvm_bitwriter.ml - LLVM Ocaml Interface ----------------*- C++ -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===
 *
 * This interface provides an ocaml API for the LLVM intermediate
 * representation, the classes in the VMCore library.
 *
 *===----------------------------------------------------------------------===*)


(* Writes the bitcode for module the given path. Returns true if successful. *)
external write_bitcode_file : Llvm.llmodule -> string -> bool
                            = "llvm_write_bitcode_file"
