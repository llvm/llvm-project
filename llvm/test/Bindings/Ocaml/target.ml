(* RUN: %ocamlc -warn-error A llvm.cma llvm_target.cma %s -o %t 2> /dev/null
 *)

(* Note: It takes several seconds for ocamlc to link an executable with
         libLLVMCore.a, so it's better to write a big test than a bunch of
         little ones. *)

open Llvm
open Llvm_target


(* Tiny unit test framework - really just to help find which line is busted *)
let suite name f =
  prerr_endline (name ^ ":");
  f ()


(*===-- Fixture -----------------------------------------------------------===*)

let filename = Sys.argv.(1)
let m = create_module filename


(*===-- Target Data -------------------------------------------------------===*)

let test_target_data () =
  let td = TargetData.create (target_triple m) in
  let sty = struct_type [| i32_type; i64_type |] in
  
  ignore (TargetData.as_string td);
  ignore (TargetData.invalidate_struct_layout td sty);
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
  
  TargetData.dispose td


(*===-- Driver ------------------------------------------------------------===*)

let _ =
  suite "target data" test_target_data;
  dispose_module m
