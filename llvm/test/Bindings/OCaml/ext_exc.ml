(* RUN: cp %s %T/ext_exc.ml
 * RUN: %ocamlc -g -w +A -package llvm.bitreader -linkpkg %T/ext_exc.ml -o %t
 * RUN: %t
 * RUN: %ocamlopt -g -w +A -package llvm.bitreader -linkpkg %T/ext_exc.ml -o %t
 * RUN: %t
 * XFAIL: vg_leak
 *)

let context = Llvm.global_context ()

let diagnostic_handler _ = ()

(* This used to crash, we must not use 'external' in .mli files, but 'val' if we
 * want the let _ bindings executed, see http://caml.inria.fr/mantis/view.php?id=4166 *)
let _ =
    Llvm.set_diagnostic_handler context (Some diagnostic_handler);
    try
        ignore (Llvm_bitreader.get_module context (Llvm.MemoryBuffer.of_stdin ()))
    with
    Llvm_bitreader.Error _ -> ();;
let _ =
    try
        ignore (Llvm.MemoryBuffer.of_file "/path/to/nonexistent/file")
    with
    Llvm.IoError _ -> ();;
