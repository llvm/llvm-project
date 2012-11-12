(* RUN: %ocamlopt -warn-error A llvm.cmxa llvm_target.cmxa %s -o %t
 * RUN: %t %t.bc
 * XFAIL: vg_leak
 *)

(* Note: It takes several seconds for ocamlopt to link an executable with
         libLLVMCore.a, so it's better to write a big test than a bunch of
         little ones. *)

open Llvm
open Llvm_target


let context = global_context ()
let i32_type = Llvm.i32_type context
let i64_type = Llvm.i64_type context

(* Tiny unit test framework - really just to help find which line is busted *)
let print_checkpoints = false

let suite name f =
  if print_checkpoints then
    prerr_endline (name ^ ":");
  f ()


(*===-- Fixture -----------------------------------------------------------===*)

let filename = Sys.argv.(1)
let m = create_module context filename


(*===-- Target Data -------------------------------------------------------===*)

let test_target_data () =
  let td = DataLayout.create (target_triple m) in
  let sty = struct_type context [| i32_type; i64_type |] in
  
  ignore (DataLayout.as_string td);
  ignore (byte_order td);
  ignore (pointer_size td);
  ignore (intptr_type td);
  ignore (size_in_bits td sty);
  ignore (store_size td sty);
  ignore (abi_size td sty);
  ignore (stack_align td sty);
  ignore (preferred_align td sty);
  ignore (preferred_align_of_global td (declare_global sty "g" m));
  ignore (element_at_offset td sty (Int64.of_int 1));
  ignore (offset_of_element td sty 1);
  
  DataLayout.dispose td


(*===-- Driver ------------------------------------------------------------===*)

let _ =
  suite "target data" test_target_data;
  dispose_module m
