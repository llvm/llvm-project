(* RUN: %ocamlc -warn-error A llvm.cma llvm_scalar_opts.cma llvm_target.cma %s -o %t 2> /dev/null
 *)

(* Note: It takes several seconds for ocamlc to link an executable with
         libLLVMCore.a, so it's better to write a big test than a bunch of
         little ones. *)

open Llvm
open Llvm_scalar_opts
open Llvm_target


(* Tiny unit test framework - really just to help find which line is busted *)
let suite name f =
  prerr_endline (name ^ ":");
  f ()


(*===-- Fixture -----------------------------------------------------------===*)

let filename = Sys.argv.(1)
let m = create_module filename
let mp = ModuleProvider.create m


(*===-- Transforms --------------------------------------------------------===*)

let test_transforms () =
  let (++) x f = ignore (f x); x in

  let fty = function_type void_type [| |] in
  let fn = define_function "fn" fty m in
  ignore (build_ret_void (builder_at_end (entry_block fn)));
  
  let td = TargetData.create (target_triple m) in
  
  ignore (PassManager.create_function mp
           ++ TargetData.add td
           ++ add_instruction_combining
           ++ add_reassociation
           ++ add_gvn
           ++ add_cfg_simplification
           ++ add_constant_propagation
           ++ PassManager.initialize
           ++ PassManager.run_function fn
           ++ PassManager.finalize
           ++ PassManager.dispose);
  
  TargetData.dispose td


(*===-- Driver ------------------------------------------------------------===*)

let _ =
  suite "transforms" test_transforms;
  ModuleProvider.dispose mp
