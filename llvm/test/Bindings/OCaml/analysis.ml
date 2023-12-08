(* RUN: rm -rf %t && mkdir -p %t && cp %s %t/analysis.ml
 * RUN: %ocamlc -g -w +A -package llvm.analysis -linkpkg %t/analysis.ml -o %t/executable
 * RUN: %t/executable
 * RUN: %ocamlopt -g -w +A -package llvm.analysis -linkpkg %t/analysis.ml -o %t/executable
 * RUN: %t/executable
 * XFAIL: vg_leak
 *)

open Llvm
open Llvm_analysis

(* Note that this takes a moment to link, so it's best to keep the number of
   individual tests low. *)

let context = global_context ()

let test x = if not x then exit 1 else ()

let bomb msg =
  prerr_endline msg;
  exit 2

let _ =
  let fty = function_type (void_type context) [| |] in
  let m = create_module context "valid_m" in
  let fn = define_function "valid_fn" fty m in
  let at_entry = builder_at_end context (entry_block fn) in
  ignore (build_ret_void at_entry);


  (* Test that valid constructs verify. *)
  begin match verify_module m with
    Some msg -> bomb "valid module failed verification!"
  | None -> ()
  end;

  if not (verify_function fn) then bomb "valid function failed verification!";


  (* Test that invalid constructs do not verify.
     A basic block can contain only one terminator instruction. *)
  ignore (build_ret_void at_entry);

  begin match verify_module m with
    Some msg -> ()
  | None -> bomb "invalid module passed verification!"
  end;

  if verify_function fn then bomb "invalid function passed verification!";


  dispose_module m

  (* Don't bother to test assert_valid_{module,function}. *)
