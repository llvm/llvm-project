(* RUN: %ocamlopt -warn-error A llvm.cmxa llvm_target.cmxa llvm_executionengine.cmxa %s -o %t
 * RUN: %t
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

let bomb msg =
  prerr_endline msg;
  exit 2

let define_main_fn m retval =
  let fn =
    let str_arr_type = pointer_type (pointer_type i8_type) in
    define_function "main" (function_type i32_type [| i32_type;
                                                      str_arr_type;
                                                      str_arr_type |]) m in
  let b = builder_at_end (global_context ()) (entry_block fn) in
  ignore (build_ret (const_int i32_type retval) b);
  fn

let define_plus m =
  let fn = define_function "plus" (function_type i32_type [| i32_type;
                                                             i32_type |]) m in
  let b = builder_at_end (global_context ()) (entry_block fn) in
  let add = build_add (param fn 0) (param fn 1) "sum" b in
  ignore (build_ret add b)

let test_genericvalue () =
  let tu = (1, 2) in
  let ptrgv = GenericValue.of_pointer tu in
  assert (tu = GenericValue.as_pointer ptrgv);
  
  let fpgv = GenericValue.of_float double_type 2. in
  assert (2. = GenericValue.as_float double_type fpgv);
  
  let intgv = GenericValue.of_int i32_type 3 in
  assert (3  = GenericValue.as_int intgv);
  
  let i32gv = GenericValue.of_int32 i32_type (Int32.of_int 4) in
  assert ((Int32.of_int 4) = GenericValue.as_int32 i32gv);
  
  let nigv = GenericValue.of_nativeint i32_type (Nativeint.of_int 5) in
  assert ((Nativeint.of_int 5) = GenericValue.as_nativeint nigv);
  
  let i64gv = GenericValue.of_int64 i64_type (Int64.of_int 6) in
  assert ((Int64.of_int 6) = GenericValue.as_int64 i64gv)

let test_executionengine () =
  (* create *)
  let m = create_module (global_context ()) "test_module" in
  let main = define_main_fn m 42 in
  
  let m2 = create_module (global_context ()) "test_module2" in
  define_plus m2;
  
  let ee = ExecutionEngine.create m in
  ExecutionEngine.add_module m2 ee;
  
  (* run_static_ctors *)
  ExecutionEngine.run_static_ctors ee;
  
  (* run_function_as_main *)
  let res = ExecutionEngine.run_function_as_main main [|"test"|] [||] ee in
  if 42 != res then bomb "main did not return 42";
  
  (* free_machine_code *)
  ExecutionEngine.free_machine_code main ee;
  
  (* find_function *)
  match ExecutionEngine.find_function "dne" ee with
  | Some _ -> raise (Failure "find_function 'dne' failed")
  | None ->
  
  match ExecutionEngine.find_function "plus" ee with
  | None -> raise (Failure "find_function 'plus' failed")
  | Some plus ->
  
  (* run_function *)
  let res = ExecutionEngine.run_function plus
                                         [| GenericValue.of_int i32_type 2;
                                            GenericValue.of_int i32_type 2 |]
                                         ee in
  if 4 != GenericValue.as_int res then bomb "plus did not work";
  
  (* remove_module *)
  Llvm.dispose_module (ExecutionEngine.remove_module m2 ee);
  
  (* run_static_dtors *)
  ExecutionEngine.run_static_dtors ee;

  (* Show that the target data binding links and runs.*)
  let td = ExecutionEngine.target_data ee in

  (* Demonstrate that a garbage pointer wasn't returned. *)
  let ty = intptr_type td in
  if ty != i32_type && ty != i64_type then bomb "target_data did not work";
  
  (* dispose *)
  ExecutionEngine.dispose ee

let _ =
  test_genericvalue ();
  test_executionengine ()
