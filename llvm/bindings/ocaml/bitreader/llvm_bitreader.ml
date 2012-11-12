(*===-- llvm_bitreader.ml - LLVM Ocaml Interface ----------------*- C++ -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)


exception Error of string

external register_exns : exn -> unit = "llvm_register_bitreader_exns"
let _ = register_exns (Error "")

external get_module : Llvm.llcontext -> Llvm.llmemorybuffer -> Llvm.llmodule
                    = "llvm_get_module"

external parse_bitcode : Llvm.llcontext -> Llvm.llmemorybuffer -> Llvm.llmodule
                       = "llvm_parse_bitcode"
