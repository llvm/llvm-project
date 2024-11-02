(* RUN: rm -rf %t && mkdir -p %t && cp %s %t/executionengine.ml
 * RUN: %ocamlc -g -w +A -thread -package llvm.executionengine -linkpkg %t/executionengine.ml -o %t/executable
 * RUN: %t/executable
 * RUN: %ocamlopt -g -w +A -thread -package llvm.executionengine -linkpkg %t/executionengine.ml -o %t/executable
 * RUN: %t/executable
 * REQUIRES: native
 * XFAIL: vg_leak
 *)

open Llvm
open Llvm_executionengine
open Llvm_target

(* Note that this takes a moment to link, so it's best to keep the number of
   individual tests low. *)

let context = global_context ()
let i8_type = Llvm.i8_type context
let i32_type = Llvm.i32_type context
let i64_type = Llvm.i64_type context
let double_type = Llvm.double_type context

let () =
  assert (Llvm_executionengine.initialize ())

let bomb msg =
  prerr_endline msg;
  exit 2

let define_getglobal m pg =
  let fty = function_type i32_type [||] in
  let fn = define_function "getglobal" fty m in
  let b = builder_at_end (global_context ()) (entry_block fn) in
  let g = build_call fty pg [||] "" b in
  ignore (build_ret g b);
  fn

let define_plus m =
  let fn = define_function "plus" (function_type i32_type [| i32_type;
                                                             i32_type |]) m in
  let b = builder_at_end (global_context ()) (entry_block fn) in
  let add = build_add (param fn 0) (param fn 1) "sum" b in
  ignore (build_ret add b);
  fn

let test_executionengine () =
  let open Ctypes in

  (* create *)
  let m = create_module (global_context ()) "test_module" in
  let ee = create m in

  (* add plus *)
  ignore (define_plus m);

  (* declare global variable *)
  ignore (define_global "globvar" (const_int i32_type 23) m);

  (* add module *)
  let m2 = create_module (global_context ()) "test_module2" in
  add_module m2 ee;

  (* add global mapping *)
  (* BROKEN: see PR20656 *)
  (* let g = declare_function "g" (function_type i32_type [||]) m2 in
  let cg = coerce (Foreign.funptr (void @-> returning int32_t)) (ptr void)
                                  (fun () -> 42l) in
  add_global_mapping g cg ee;

  (* check g *)
  let cg' = get_pointer_to_global g (ptr void) ee in
  if 0 <> ptr_compare cg cg' then bomb "int pointers to g differ";

  (* add getglobal *)
  let getglobal = define_getglobal m2 g in*)

  (* run_static_ctors *)
  run_static_ctors ee;

  (* get a handle on globvar *)
  let varh    = get_global_value_address "globvar" int32_t ee in
  if 23l <> varh then bomb "get_global_value_address didn't work";

  (* call plus *)
  let cplusty = Foreign.funptr (int32_t @-> int32_t @-> returning int32_t) in
  let cplus   = get_function_address "plus" cplusty ee in
  if 4l <> cplus 2l 2l then bomb "plus didn't work";

  (* call getglobal *)
  (* let cgetglobalty = Foreign.funptr (void @-> returning int32_t) in
  let cgetglobal   = get_pointer_to_global getglobal cgetglobalty ee in
  if 42l <> cgetglobal () then bomb "getglobal didn't work"; *)

  (* remove_module *)
  remove_module m2 ee;
  dispose_module m2;

  (* run_static_dtors *)
  run_static_dtors ee;

  (* Show that the data layout binding links and runs.*)
  let dl = data_layout ee in

  (* Demonstrate that a garbage pointer wasn't returned. *)
  let ty = DataLayout.intptr_type context dl in
  if ty != i32_type && ty != i64_type then bomb "target_data did not work";

  (* dispose *)
  dispose ee

let () =
  test_executionengine ();
  Gc.compact ()
