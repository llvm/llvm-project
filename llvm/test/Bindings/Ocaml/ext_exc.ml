(* RUN: %ocamlopt -warn-error A llvm.cmxa llvm_bitreader.cmxa llvm_executionengine.cmxa %s -o %t
 * RUN: %t </dev/null
 * XFAIL: vg_leak
 *)
let context = Llvm.global_context ()
(* this used to crash, we must not use 'external' in .mli files, but 'val' if we
 * want the let _ bindings executed, see http://caml.inria.fr/mantis/view.php?id=4166 *)
let _ =
    try
        ignore (Llvm_bitreader.get_module context (Llvm.MemoryBuffer.of_stdin ()))
    with
    Llvm_bitreader.Error _ -> ();;
let _ =
    try
        ignore (Llvm.MemoryBuffer.of_file "/path/to/nonexistent/file")
    with
    Llvm.IoError _ -> ();;
