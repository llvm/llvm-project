(*===-- llvm_bitreader.mli - LLVM Ocaml Interface ---------------*- C++ -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

(** Bitcode reader.

    This interface provides an ocaml API for the LLVM bitcode reader, the
    classes in the Bitreader library. *)

exception Error of string

(** [read_bitcode_file path] reads the bitcode for a new module [m] from the
    file at [path]. Returns [Success m] if successful, and [Failure msg]
    otherwise, where [msg] is a description of the error encountered.
    See the function [llvm::getBitcodeModuleProvider]. *)
external get_module_provider : Llvm.llmemorybuffer -> Llvm.llmoduleprovider
                             = "llvm_get_module_provider"

(** [parse_bitcode mb] parses the bitcode for a new module [m] from the memory
    buffer [mb]. Returns [Success m] if successful, and [Failure msg] otherwise,
    where [msg] is a description of the error encountered.
    See the function [llvm::ParseBitcodeFile]. *)
external parse_bitcode : Llvm.llmemorybuffer -> Llvm.llmodule
                       = "llvm_parse_bitcode"
