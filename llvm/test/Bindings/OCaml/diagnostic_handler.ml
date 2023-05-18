(* RUN: rm -rf %t && mkdir -p %t && cp %s %t/diagnostic_handler.ml
 * RUN: %ocamlc -g -w +A -package llvm.bitreader -linkpkg %t/diagnostic_handler.ml -o %t/executable
 * RUN: %t/executable %t/bitcode.bc | FileCheck %s
 * RUN: %ocamlopt -g -w +A -package llvm.bitreader -linkpkg %t/diagnostic_handler.ml -o %t/executable
 * RUN: %t/executable %t/bitcode.bc | FileCheck %s
 * XFAIL: vg_leak
 *)

let context = Llvm.global_context ()

let diagnostic_handler d =
  Printf.printf
    "Diagnostic handler called: %s\n" (Llvm.Diagnostic.description d);
  match Llvm.Diagnostic.severity d with
  | Error -> Printf.printf "Diagnostic severity is Error\n"
  | Warning -> Printf.printf "Diagnostic severity is Warning\n"
  | Remark -> Printf.printf "Diagnostic severity is Remark\n"
  | Note -> Printf.printf "Diagnostic severity is Note\n"

let test x = if not x then exit 1 else ()

let _ =
  Llvm.set_diagnostic_handler context (Some diagnostic_handler);

  (* corrupt the bitcode *)
  let fn = Sys.argv.(1) ^ ".txt" in
  begin let oc = open_out fn in
    output_string oc "not a bitcode file\n";
    close_out oc
  end;

  test begin
    try
      let mb = Llvm.MemoryBuffer.of_file fn in
      let m = begin try
        (* CHECK: Diagnostic handler called: Invalid bitcode signature
         * CHECK: Diagnostic severity is Error
         *)
        Llvm_bitreader.get_module context mb
      with x ->
        Llvm.MemoryBuffer.dispose mb;
        raise x
      end in
      Llvm.dispose_module m;
      false
    with Llvm_bitreader.Error _ ->
      true
  end
