(* RUN: rm -rf %t && mkdir -p %t && cp %s %t/ipo_opts.ml
 * RUN: %ocamlc -g -w +A -package llvm.ipo -linkpkg %t/ipo_opts.ml -o %t/executable
 * RUN: %t/executable %t/bitcode.bc
 * RUN: %ocamlopt -g -w +A -package llvm.ipo -linkpkg %t/ipo_opts.ml -o %t/executable
 * RUN: %t/executable %t/bitcode.bc
 * XFAIL: vg_leak
 *)

(* Note: It takes several seconds for ocamlopt to link an executable with
         libLLVMCore.a, so it's better to write a big test than a bunch of
         little ones. *)

open Llvm
open Llvm_ipo
open Llvm_target

let context = global_context ()
let void_type = Llvm.void_type context
let i8_type = Llvm.i8_type context

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

  let fty = function_type i8_type [| |] in
  let fn = define_function "fn" fty m in
  let fn2 = define_function "fn2" fty m in begin
      ignore (build_ret (const_int i8_type 4) (builder_at_end context (entry_block fn)));
      let b = builder_at_end context  (entry_block fn2) in
      ignore (build_ret (build_call fty fn [| |] "" b) b);
  end;

  ignore (PassManager.create ()
           ++ add_constant_merge
           ++ add_dead_arg_elimination
           ++ add_function_attrs
           ++ add_function_inlining
           ++ add_always_inliner
           ++ add_global_dce
           ++ add_global_optimizer
           ++ add_ipsccp
           ++ add_internalize ~all_but_main:true
           ++ add_strip_dead_prototypes
           ++ add_strip_symbols
           ++ PassManager.run_module m
           ++ PassManager.dispose)


(*===-- Driver ------------------------------------------------------------===*)

let _ =
  suite "transforms" test_transforms;
  dispose_module m
