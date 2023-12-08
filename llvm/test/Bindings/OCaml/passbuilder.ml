(* RUN: rm -rf %t && mkdir -p %t && cp %s %t/passbuilder.ml
 * RUN: %ocamlc -g -w +A -package llvm.passbuilder -package llvm.all_backends -linkpkg %t/passbuilder.ml -o %t/executable
 * RUN: %t/executable
 * RUN: %ocamlopt -g -w +A -package llvm.passbuilder -package llvm.all_backends -linkpkg %t/passbuilder.ml -o %t/executable
 * RUN: %t/executable
 * XFAIL: vg_leak
 *)

let () = Llvm_all_backends.initialize ()

(*===-- Fixture -----------------------------------------------------------===*)

let context = Llvm.global_context ()

let m = Llvm.create_module context "mymodule"

let () =
  let ty =
    Llvm.function_type (Llvm.void_type context)
      [| Llvm.i1_type context;
         Llvm.pointer_type context;
         Llvm.pointer_type context |]
  in
  let foo = Llvm.define_function "foo" ty m in
  let entry = Llvm.entry_block foo in
  let builder = Llvm.builder_at_end context entry in
  ignore
    (Llvm.build_store
       (Llvm.const_int (Llvm.i8_type context) 42) (Llvm.param foo 1) builder);
  let loop = Llvm.append_block context "loop" foo in
  Llvm.position_at_end loop builder;
  ignore
    (Llvm.build_load (Llvm.i8_type context) (Llvm.param foo 2) "tmp1" builder);
  ignore (Llvm.build_br loop builder);
  let exit = Llvm.append_block context "exit" foo in
  Llvm.position_at_end exit builder;
  ignore (Llvm.build_ret_void builder);
  Llvm.position_at_end entry builder;
  ignore (Llvm.build_cond_br (Llvm.param foo 0) loop exit builder)

let target =
  Llvm_target.Target.by_triple (Llvm_target.Target.default_triple ())

let machine =
  Llvm_target.TargetMachine.create
    ~triple:(Llvm_target.Target.default_triple ()) target

let options = Llvm_passbuilder.create_passbuilder_options ()

(*===-- PassBuilder -------------------------------------------------------===*)
let () =
  Llvm_passbuilder.passbuilder_options_set_verify_each options true;
  Llvm_passbuilder.passbuilder_options_set_debug_logging options true;
  Llvm_passbuilder.passbuilder_options_set_loop_interleaving options true;
  Llvm_passbuilder.passbuilder_options_set_loop_vectorization options true;
  Llvm_passbuilder.passbuilder_options_set_slp_vectorization options true;
  Llvm_passbuilder.passbuilder_options_set_loop_unrolling options true;
  Llvm_passbuilder.passbuilder_options_set_forget_all_scev_in_loop_unroll
    options true;
  Llvm_passbuilder.passbuilder_options_set_licm_mssa_opt_cap options 2;
  Llvm_passbuilder.passbuilder_options_set_licm_mssa_no_acc_for_promotion_cap
    options 2;
  Llvm_passbuilder.passbuilder_options_set_call_graph_profile options true;
  Llvm_passbuilder.passbuilder_options_set_merge_functions options true;
  Llvm_passbuilder.passbuilder_options_set_inliner_threshold options 2;
  match Llvm_passbuilder.run_passes m "no-op-module" machine options with
  | Error e ->
    prerr_endline e;
    assert false
  | Ok () -> ()

let () =
  Llvm_passbuilder.dispose_passbuilder_options options;
  Llvm.dispose_module m
