(* RUN: rm -rf %t && mkdir -p %t && cp %s %t/irreader.ml
 * RUN: %ocamlc -g -w +A -package llvm.irreader -linkpkg %t/irreader.ml -o %t/executable
 * RUN: %t/executable
 * RUN: %ocamlopt -g -w +A -package llvm.irreader -linkpkg %t/irreader.ml -o %t/executable
 * RUN: %t/executable
 * XFAIL: vg_leak
 *)

(* Note: It takes several seconds for ocamlopt to link an executable with
         libLLVMCore.a, so it's better to write a big test than a bunch of
         little ones. *)

open Llvm
open Llvm_irreader

let context = create_context ()

(* Tiny unit test framework - really just to help find which line is busted *)
let print_checkpoints = false

let suite name f =
  if print_checkpoints then
    prerr_endline (name ^ ":");
  f ()

let _ =
  Printexc.record_backtrace true

let insist cond =
  if not cond then failwith "insist"

(* TODO: Replace with Fun.protect when the minimum OCaml version supports it. *)
let protect ~finally f =
  try
    let r = f () in
    finally ();
    r
  with x ->
    finally ();
    raise x


(*===-- IR Reader ---------------------------------------------------------===*)

let test_irreader () =
  begin
    let buf = MemoryBuffer.of_string "@foo = global i32 42" in
    let m = protect ~finally:(fun () -> MemoryBuffer.dispose buf)
              (fun () -> parse_ir_bitcode_or_assembly context buf) in
    protect ~finally:(fun () -> dispose_module m) (fun () ->
      match lookup_global "foo" m with
      | Some foo ->
          insist (global_initializer foo =
                  Some (const_int (i32_type context) 42))
      | None ->
          failwith "global")
  end;

  begin
    let buf = MemoryBuffer.of_string "@foo = global garble" in
    let parsed = protect ~finally:(fun () -> MemoryBuffer.dispose buf)
                   (fun () ->
                     try
                       let m = parse_ir_bitcode_or_assembly context buf in
                       dispose_module m;
                       true
                     with Llvm_irreader.Error _ ->
                       false)
    in
    if parsed then
      failwith "parsed"
  end


(*===-- Opcodes -----------------------------------------------------------===*)

(* [instr_opcode] must map every opcode of the C API, including the ones added
   after [Freeze]: [PtrToAddr], [UncondBr], [CondBr], [BitInsert] and
   [BitExtract]. *)
let test_opcodes () =
  let ir = {|
define b32 @f(ptr %p, i1 %c, b32 %b, i8 %v) {
entry:
  %a = ptrtoaddr ptr %p to i64
  br label %next
next:
  %i = bitinsert b32 %b, i8 %v, i32 3
  %e = bitextract i8, b32 %i, i32 24
  br i1 %c, label %next, label %exit
exit:
  %f = freeze b32 %i
  ret b32 %f
}
|} in
  let buf = MemoryBuffer.of_string ir in
  let m = protect ~finally:(fun () -> MemoryBuffer.dispose buf)
            (fun () -> parse_ir_bitcode_or_assembly context buf) in
  protect ~finally:(fun () -> dispose_module m) (fun () ->
    match lookup_function "f" m with
    | None -> failwith "function"
    | Some f ->
        let opcodes =
          fold_left_blocks
            (fun acc bb -> fold_left_instrs (fun acc i -> instr_opcode i :: acc) acc bb)
            [] f
          |> List.rev
        in
        insist (opcodes = Opcode.[ PtrToAddr; UncondBr; BitInsert; BitExtract;
                                   CondBr; Freeze; Ret ]))


(*===-- Driver ------------------------------------------------------------===*)

let _ =
  suite "irreader" test_irreader;
  suite "opcodes" test_opcodes;
  dispose_context context
