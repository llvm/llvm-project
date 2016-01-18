(*===-- llvm_linker.ml - LLVM OCaml Interface ------------------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

exception Error of string

let () = Callback.register_exception "Llvm_linker.Error" (Error "")

external link_modules : Llvm.llmodule -> Llvm.llmodule -> unit
                      = "llvm_link_modules"
