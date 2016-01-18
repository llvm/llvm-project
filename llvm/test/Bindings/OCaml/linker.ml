(* RUN: cp %s %T/linker.ml
 * RUN: %ocamlc -g -w +A -package llvm.linker -linkpkg %T/linker.ml -o %t
 * RUN: %t
 * RUN: %ocamlopt -g -w +A -package llvm.linker -linkpkg %T/linker.ml -o %t
 * RUN: %t
 * XFAIL: vg_leak
 *)

(* Note: It takes several seconds for ocamlopt to link an executable with
         libLLVMCore.a, so it's better to write a big test than a bunch of
         little ones. *)

open Llvm
open Llvm_linker

let context = global_context ()
let void_type = Llvm.void_type context

(* Tiny unit test framework - really just to help find which line is busted *)
let print_checkpoints = false

let suite name f =
  if print_checkpoints then
    prerr_endline (name ^ ":");
  f ()


(*===-- Linker -----------------------------------------------------------===*)

let test_linker () =
  let fty = function_type void_type [| |] in

  let make_module name =
    let m = create_module context name in
    let fn = define_function ("fn_" ^ name) fty m in
    ignore (build_ret_void (builder_at_end context (entry_block fn)));
    m
  in

  let m1 = make_module "one"
  and m2 = make_module "two" in
  link_modules m1 m2;
  dispose_module m1;
  dispose_module m2;

  let m1 = make_module "one"
  and m2 = make_module "two" in
  link_modules m1 m2;
  dispose_module m1;

  let m1 = make_module "one"
  and m2 = make_module "one" in
  try
    link_modules m1 m2;
    failwith "must raise"
  with Error _ ->
    dispose_module m1;
    dispose_module m2

(*===-- Driver ------------------------------------------------------------===*)

let _ =
  suite "linker" test_linker
