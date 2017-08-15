(* RUN: rm -rf %t && mkdir -p %t && cp %s %t/bitwriter.ml
 * RUN: %ocamlc -g -w -3 -w +A -package llvm.bitreader -package llvm.bitwriter -linkpkg %t/bitwriter.ml -o %t/executable
 * RUN: %t/executable %t/bitcode.bc
 * RUN: %ocamlopt -g -w -3 -w +A -package llvm.bitreader -package llvm.bitwriter -linkpkg %t/bitwriter.ml -o %t/executable
 * RUN: %t/executable %t/bitcode.bc
 * RUN: llvm-dis < %t/bitcode.bc
 * XFAIL: vg_leak
 *)

(* Note that this takes a moment to link, so it's best to keep the number of
   individual tests low. *)

let context = Llvm.global_context ()

let test x = if not x then exit 1 else ()

let read_file name =
  let ic = open_in_bin name in
  let len = in_channel_length ic in
  let buf = String.create len in

  test ((input ic buf 0 len) = len);

  close_in ic;

  buf

let temp_bitcode ?unbuffered m =
  let temp_name, temp_oc = Filename.open_temp_file ~mode:[Open_binary] "" "" in

  test (Llvm_bitwriter.output_bitcode ?unbuffered temp_oc m);
  flush temp_oc;

  let temp_buf = read_file temp_name in

  close_out temp_oc;

  temp_buf

let _ =
  let m = Llvm.create_module context "ocaml_test_module" in

  test (Llvm_bitwriter.write_bitcode_file m Sys.argv.(1));
  let file_buf = read_file Sys.argv.(1) in

  test (file_buf = temp_bitcode m);
  test (file_buf = temp_bitcode ~unbuffered:false m);
  test (file_buf = temp_bitcode ~unbuffered:true m);
  test (file_buf = Llvm.MemoryBuffer.as_string (Llvm_bitwriter.write_bitcode_to_memory_buffer m))
