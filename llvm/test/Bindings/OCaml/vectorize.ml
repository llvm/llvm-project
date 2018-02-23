(* RUN: cp %s %T/vectorize_opts.ml
 * RUN: %ocamlc -g -w +A -package llvm.vectorize -linkpkg %T/vectorize_opts.ml -o %t
 * RUN: %t %t.bc
 * RUN: %ocamlopt -g -w +A -package llvm.vectorize -linkpkg %T/vectorize_opts.ml -o %t
 * RUN: %t %t.bc
 * XFAIL: vg_leak
 *)

(* Note: It takes several seconds for ocamlopt to link an executable with
         libLLVMCore.a, so it's better to write a big test than a bunch of
         little ones. *)

open Llvm
open Llvm_vectorize
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

  ignore (PassManager.create ()
           ++ add_bb_vectorize
           ++ add_loop_vectorize
           ++ add_slp_vectorize
           ++ PassManager.run_module m
           ++ PassManager.dispose)


(*===-- Driver ------------------------------------------------------------===*)

let _ =
  suite "transforms" test_transforms;
  dispose_module m
