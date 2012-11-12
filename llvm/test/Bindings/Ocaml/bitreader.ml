(* RUN: %ocamlopt -warn-error A llvm.cmxa llvm_bitreader.cmxa llvm_bitwriter.cmxa %s -o %t
 * RUN: %t %t.bc
 * RUN: llvm-dis < %t.bc
 * XFAIL: vg_leak
 *)

(* Note that this takes a moment to link, so it's best to keep the number of
   individual tests low. *)

let context = Llvm.global_context ()

let test x = if not x then exit 1 else ()

let _ =
  let fn = Sys.argv.(1) in
  let m = Llvm.create_module context "ocaml_test_module" in
  
  test (Llvm_bitwriter.write_bitcode_file m fn);
  
  Llvm.dispose_module m;
  
  (* parse_bitcode *)
  begin
    let mb = Llvm.MemoryBuffer.of_file fn in
    begin try
      let m = Llvm_bitreader.parse_bitcode context mb in
      Llvm.dispose_module m
    with x ->
      Llvm.MemoryBuffer.dispose mb;
      raise x
    end
  end;
  
  (* MemoryBuffer.of_file *)
  test begin try
    let mb = Llvm.MemoryBuffer.of_file (fn ^ ".bogus") in
    Llvm.MemoryBuffer.dispose mb;
    false
  with Llvm.IoError _ ->
    true
  end;
  
  (* get_module *)
  begin
    let mb = Llvm.MemoryBuffer.of_file fn in
    let m = begin try
      Llvm_bitreader.get_module context mb
    with x ->
      Llvm.MemoryBuffer.dispose mb;
      raise x
    end in
    Llvm.dispose_module m
  end;
  
  (* corrupt the bitcode *)
  let fn = fn ^ ".txt" in
  begin let oc = open_out fn in
    output_string oc "not a bitcode file\n";
    close_out oc
  end;
  
  (* test get_module exceptions *)
  test begin
    try
      let mb = Llvm.MemoryBuffer.of_file fn in
      let m = begin try
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
