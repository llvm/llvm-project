(*===-- llvm_scalar_opts.ml - LLVM Ocaml Interface -------------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

external add_constant_propagation : [<Llvm.PassManager.any] Llvm.PassManager.t
                                    -> unit
                                  = "llvm_add_constant_propagation"
external add_instruction_combining : [<Llvm.PassManager.any] Llvm.PassManager.t
                                     -> unit
                                   = "llvm_add_instruction_combining"
external
add_memory_to_register_promotion : [<Llvm.PassManager.any] Llvm.PassManager.t
                                   -> unit
                                 = "llvm_add_memory_to_register_promotion"
external
add_memory_to_register_demotion : [<Llvm.PassManager.any] Llvm.PassManager.t
                                  -> unit
                                = "llvm_add_memory_to_register_demotion"
external add_reassociation : [<Llvm.PassManager.any] Llvm.PassManager.t
                             -> unit
                           = "llvm_add_reassociation"
external add_gvn : [<Llvm.PassManager.any] Llvm.PassManager.t
                   -> unit
                 = "llvm_add_gvn"
external add_cfg_simplification : [<Llvm.PassManager.any] Llvm.PassManager.t
                                  -> unit
                                = "llvm_add_cfg_simplification"
