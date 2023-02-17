(* RUN: rm -rf %t && mkdir -p %t && cp %s %t/scalar_opts.ml
 * RUN: %ocamlc -g -w +A -package llvm.scalar_opts -linkpkg %t/scalar_opts.ml -o %t/executable
 * RUN: %t/executable %t/bitcode.bc
 * RUN: %ocamlopt -g -w +A -package llvm.scalar_opts -linkpkg %t/scalar_opts.ml -o %t/executable
 * RUN: %t/executable %t/bitcode.bc
 * XFAIL: vg_leak
 *)

(* Note: It takes several seconds for ocamlopt to link an executable with
         libLLVMCore.a, so it's better to write a big test than a bunch of
         little ones. *)

open Llvm
open Llvm_scalar_opts
open Llvm_target

let context = global_context ()
let void_type = Llvm.void_type context

(* Tiny unit test framework - really just to help find which line is busted *)
let print_checkpoints = false

let suite name f =
  if print_checkpoints then
    prerr_endline (name ^ ":");
  f ()


(*===-- Fixture -----------------------------------------------------------===*)

let filename = Sys.argv.(1)
let m = create_module context filename


(*===-- Transforms --------------------------------------------------------===*)

let test_transforms () =
  let (++) x f = f x; x in

  let fty = function_type void_type [| |] in
  let fn = define_function "fn" fty m in
  ignore (build_ret_void (builder_at_end context (entry_block fn)));

  ignore (PassManager.create_function m
           ++ add_aggressive_dce
           ++ add_alignment_from_assumptions
           ++ add_cfg_simplification
           ++ add_dead_store_elimination
           ++ add_scalarizer
           ++ add_merged_load_store_motion
           ++ add_gvn
           ++ add_ind_var_simplification
           ++ add_instruction_combination
           ++ add_jump_threading
           ++ add_licm
           ++ add_loop_idiom
           ++ add_loop_rotation
           ++ add_loop_unroll
           ++ add_memcpy_opt
           ++ add_partially_inline_lib_calls
           ++ add_lower_switch
           ++ add_memory_to_register_promotion
           ++ add_reassociation
           ++ add_sccp
           ++ add_scalar_repl_aggregation
           ++ add_scalar_repl_aggregation_ssa
           ++ add_scalar_repl_aggregation_with_threshold 4
           ++ add_lib_call_simplification
           ++ add_tail_call_elimination
           ++ add_memory_to_register_demotion
           ++ add_verifier
           ++ add_correlated_value_propagation
           ++ add_early_cse
           ++ add_lower_expect_intrinsic
           ++ add_type_based_alias_analysis
           ++ add_scoped_no_alias_alias_analysis
           ++ add_basic_alias_analysis
           ++ PassManager.initialize
           ++ PassManager.run_function fn
           ++ PassManager.finalize
           ++ PassManager.dispose)


(*===-- Driver ------------------------------------------------------------===*)

let _ =
  suite "transforms" test_transforms;
  dispose_module m
