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
external add_sccp : [<Llvm.PassManager.any] Llvm.PassManager.t -> unit
                  = "llvm_add_sccp"
external add_dead_store_elimination : [<Llvm.PassManager.any] Llvm.PassManager.t
                                      -> unit
                                    = "llvm_add_dead_store_elimination"
external add_aggressive_dce : [<Llvm.PassManager.any] Llvm.PassManager.t -> unit
                            = "llvm_add_aggressive_dce"
external
add_scalar_repl_aggregation : [<Llvm.PassManager.any] Llvm.PassManager.t -> unit
                            = "llvm_add_scalar_repl_aggregation"

external
add_scalar_repl_aggregation_ssa : [<Llvm.PassManager.any] Llvm.PassManager.t -> unit
                            = "llvm_add_scalar_repl_aggregation_ssa"

external
add_scalar_repl_aggregation_with_threshold : int -> [<Llvm.PassManager.any] Llvm.PassManager.t
                                             -> unit
                            = "llvm_add_scalar_repl_aggregation_with_threshold"
external add_ind_var_simplification : [<Llvm.PassManager.any] Llvm.PassManager.t
                                      -> unit
                                    = "llvm_add_ind_var_simplification"
external
add_instruction_combination : [<Llvm.PassManager.any] Llvm.PassManager.t
                              -> unit
                            = "llvm_add_instruction_combination"
external add_licm : [<Llvm.PassManager.any] Llvm.PassManager.t
                                -> unit
                              = "llvm_add_licm"
external add_loop_unswitch : [<Llvm.PassManager.any] Llvm.PassManager.t
                                -> unit
                              = "llvm_add_loop_unswitch"
external add_loop_unroll : [<Llvm.PassManager.any] Llvm.PassManager.t
                                -> unit
                              = "llvm_add_loop_unroll"
external add_loop_rotation : [<Llvm.PassManager.any] Llvm.PassManager.t
                             -> unit
                           = "llvm_add_loop_rotation"
external
add_memory_to_register_promotion : [<Llvm.PassManager.any] Llvm.PassManager.t
                                   -> unit
                                 = "llvm_add_memory_to_register_promotion"
external
add_memory_to_register_demotion : [<Llvm.PassManager.any] Llvm.PassManager.t
                                  -> unit
                                = "llvm_add_memory_to_register_demotion"
external add_reassociation : [<Llvm.PassManager.any] Llvm.PassManager.t -> unit
                           = "llvm_add_reassociation"
external add_jump_threading : [<Llvm.PassManager.any] Llvm.PassManager.t
                                -> unit
                              = "llvm_add_jump_threading"
external add_cfg_simplification : [<Llvm.PassManager.any] Llvm.PassManager.t
                                  -> unit
                                = "llvm_add_cfg_simplification"
external
add_tail_call_elimination : [<Llvm.PassManager.any] Llvm.PassManager.t -> unit
                          = "llvm_add_tail_call_elimination" 
external add_gvn : [<Llvm.PassManager.any] Llvm.PassManager.t
                   -> unit
                 = "llvm_add_gvn"
external add_memcpy_opt : [<Llvm.PassManager.any] Llvm.PassManager.t
                                -> unit
                              = "llvm_add_memcpy_opt"
external add_loop_deletion : [<Llvm.PassManager.any] Llvm.PassManager.t
                             -> unit
                           = "llvm_add_loop_deletion"

external add_loop_idiom : [<Llvm.PassManager.any] Llvm.PassManager.t
                             -> unit
                           = "llvm_add_loop_idiom"

external
add_lib_call_simplification : [<Llvm.PassManager.any] Llvm.PassManager.t -> unit
                            = "llvm_add_lib_call_simplification"

external
add_verifier : [<Llvm.PassManager.any] Llvm.PassManager.t -> unit
        = "llvm_add_verifier"

external
add_correlated_value_propagation : [<Llvm.PassManager.any] Llvm.PassManager.t -> unit
        = "llvm_add_correlated_value_propagation"

external
add_early_cse : [<Llvm.PassManager.any] Llvm.PassManager.t -> unit
        = "llvm_add_early_cse"

external
add_lower_expect_intrinsic : [<Llvm.PassManager.any] Llvm.PassManager.t -> unit
        = "llvm_add_lower_expect_intrinsic"

external
add_type_based_alias_analysis : [<Llvm.PassManager.any] Llvm.PassManager.t -> unit
        = "llvm_add_type_based_alias_analysis"

external
add_basic_alias_analysis : [<Llvm.PassManager.any] Llvm.PassManager.t -> unit
        = "llvm_add_basic_alias_analysis"

