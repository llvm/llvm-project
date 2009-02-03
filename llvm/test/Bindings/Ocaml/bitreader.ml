(* RUN: %ocamlc -warn-error A llvm.cma llvm_bitreader.cma llvm_bitwriter.cma %s -o %t 2> /dev/null
 * RUN: ./%t %t.bc
 * RUN: llvm-dis < %t.bc | grep caml_int_ty
 *)

(* Note that this takes a moment to link, so it's best to keep the number of
   individual tests low. *)

let test x = if not x then exit 1 else ()

let _ =
  let fn = Sys.argv.(1) in
  let m = Llvm.create_module "ocaml_test_module" in
  
  ignore (Llvm.define_type_name "caml_int_ty" Llvm.i32_type m);
  
  test (Llvm_bitwriter.write_bitcode_file m fn);
  
  Llvm.dispose_module m;
  
  (* parse_bitcode *)
  begin
    let mb = Llvm.MemoryBuffer.of_file fn in
    begin try
      let m = Llvm_bitreader.parse_bitcode mb in
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
  
  (* get_module_provider *)
  begin
    let mb = Llvm.MemoryBuffer.of_file fn in
    let mp = begin try
      Llvm_bitreader.get_module_provider mb
    with x ->
      Llvm.MemoryBuffer.dispose mb;
      raise x
    end in
    Llvm.ModuleProvider.dispose mp
  end;
  
  (* corrupt the bitcode *)
  let fn = fn ^ ".txt" in
  begin let oc = open_out fn in
    output_string oc "not a bitcode file\n";
    close_out oc
  end;
  
  (* test get_module_provider exceptions *)
  test begin
    try
      let mb = Llvm.MemoryBuffer.of_file fn in
      let mp = begin try
        Llvm_bitreader.get_module_provider mb
      with x ->
        Llvm.MemoryBuffer.dispose mb;
        raise x
      end in
      Llvm.ModuleProvider.dispose mp;
      false
    with Llvm_bitreader.Error _ ->
      true
  end
