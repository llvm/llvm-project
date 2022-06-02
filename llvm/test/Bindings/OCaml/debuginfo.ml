(* RUN: rm -rf %t && mkdir -p %t && cp %s %t/debuginfo.ml && cp %S/Utils/Testsuite.ml %t/Testsuite.ml
 * RUN: %ocamlc -g -w +A -package llvm.all_backends -package llvm.target -package llvm.analysis -package llvm.debuginfo -I %t/ -linkpkg %t/Testsuite.ml %t/debuginfo.ml -o %t/executable
 * RUN: %t/executable | FileCheck %s
 * RUN: %ocamlopt -g -w +A -package llvm.all_backends -package llvm.target -package llvm.analysis -package llvm.debuginfo -I %t/ -linkpkg %t/Testsuite.ml %t/debuginfo.ml -o %t/executable
 * RUN: %t/executable | FileCheck %s
 * XFAIL: vg_leak
 *)

open Testsuite

let context = Llvm.global_context ()

let filename = "di_test_file"

let directory = "di_test_dir"

let module_name = "di_test_module"

let null_metadata = Llvm_debuginfo.llmetadata_null ()

let string_of_metadata md =
  Llvm.string_of_llvalue (Llvm.metadata_as_value context md)

let stdout_metadata md = Printf.printf "%s\n" (string_of_metadata md)

let prepare_target llmod =
  Llvm_all_backends.initialize ();
  let triple = Llvm_target.Target.default_triple () in
  let lltarget = Llvm_target.Target.by_triple triple in
  let llmachine = Llvm_target.TargetMachine.create ~triple lltarget in
  let lldly =
    Llvm_target.DataLayout.as_string
      (Llvm_target.TargetMachine.data_layout llmachine)
  in
  let _ = Llvm.set_target_triple triple llmod in
  let _ = Llvm.set_data_layout lldly llmod in
  ()

let new_module () =
  let m = Llvm.create_module context module_name in
  let () = prepare_target m in
  m

let test_get_module () =
  group "module_level_tests";
  let m = new_module () in
  let cur_ver = Llvm_debuginfo.debug_metadata_version () in
  insist (cur_ver > 0);
  let m_ver = Llvm_debuginfo.get_module_debug_metadata_version m in
  (* We haven't added any debug info to the module *)
  insist (m_ver = 0);
  let dibuilder = Llvm_debuginfo.dibuilder m in
  let di_version_key = "Debug Info Version" in
  let ver =
    Llvm.value_as_metadata @@ Llvm.const_int (Llvm.i32_type context) cur_ver
  in
  let () =
    Llvm.add_module_flag m Llvm.ModuleFlagBehavior.Warning di_version_key ver
  in
  let file_di =
    Llvm_debuginfo.dibuild_create_file dibuilder ~filename ~directory
  in
  stdout_metadata file_di;
  (* CHECK: [[FILE_PTR:<0x[0-9a-f]*>]] = !DIFile(filename: "di_test_file", directory: "di_test_dir")
  *)
  insist
    ( Llvm_debuginfo.di_file_get_filename ~file:file_di = filename
    && Llvm_debuginfo.di_file_get_directory ~file:file_di = directory );
  insist
    ( Llvm_debuginfo.get_metadata_kind file_di
    = Llvm_debuginfo.MetadataKind.DIFileMetadataKind );
  let cu_di =
    Llvm_debuginfo.dibuild_create_compile_unit dibuilder
      Llvm_debuginfo.DWARFSourceLanguageKind.C89 ~file_ref:file_di
      ~producer:"TestGen" ~is_optimized:false ~flags:"" ~runtime_ver:0
      ~split_name:"" Llvm_debuginfo.DWARFEmissionKind.LineTablesOnly ~dwoid:0
      ~di_inlining:false ~di_profiling:false ~sys_root:"" ~sdk:""
  in
  stdout_metadata cu_di;
  (* CHECK: [[CMPUNIT_PTR:<0x[0-9a-f]*>]] = distinct !DICompileUnit(language: DW_LANG_C89, file: [[FILE_PTR]], producer: "TestGen", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false)
  *)
  insist
    ( Llvm_debuginfo.get_metadata_kind cu_di
    = Llvm_debuginfo.MetadataKind.DICompileUnitMetadataKind );
  let m_di =
    Llvm_debuginfo.dibuild_create_module dibuilder ~parent_ref:cu_di
      ~name:module_name ~config_macros:"" ~include_path:"" ~sys_root:""
  in
  insist
    ( Llvm_debuginfo.get_metadata_kind m_di
    = Llvm_debuginfo.MetadataKind.DIModuleMetadataKind );
  insist (Llvm_debuginfo.get_module_debug_metadata_version m = cur_ver);
  stdout_metadata m_di;
  (* CHECK: [[MODULE_PTR:<0x[0-9a-f]*>]] = !DIModule(scope: null, name: "di_test_module")
  *)
  (m, dibuilder, file_di, m_di)

let flags_zero = Llvm_debuginfo.diflags_get Llvm_debuginfo.DIFlag.Zero

let int_ty_di bits dibuilder =
  Llvm_debuginfo.dibuild_create_basic_type dibuilder ~name:"int"
    ~size_in_bits:bits ~encoding:0x05
    (* llvm::dwarf::DW_ATE_signed *) flags_zero

let test_get_function m dibuilder file_di m_di =
  group "function_level_tests";

  (* Create a function of type "void foo (int)". *)
  let int_ty_di = int_ty_di 32 dibuilder in
  stdout_metadata int_ty_di;
  (* CHECK: [[INT32_PTR:<0x[0-9a-f]*>]] = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
  *)
  let param_types = [| null_metadata; int_ty_di |] in
  let fty_di =
    Llvm_debuginfo.dibuild_create_subroutine_type dibuilder ~file:file_di
      ~param_types flags_zero
  in
  insist
    ( Llvm_debuginfo.get_metadata_kind fty_di
    = Llvm_debuginfo.MetadataKind.DISubroutineTypeMetadataKind );
  (* To be able to print and verify the type array of the subroutine type,
   * since we have no way to access it from fty_di, we build it again. *)
  let fty_di_args =
    Llvm_debuginfo.dibuild_get_or_create_type_array dibuilder ~data:param_types
  in
  stdout_metadata fty_di_args;
  (* CHECK: [[FARGS_PTR:<0x[0-9a-f]*>]] = !{null, [[INT32_PTR]]}
  *)
  stdout_metadata fty_di;
  (* CHECK: [[SBRTNTY_PTR:<0x[0-9a-f]*>]] = !DISubroutineType(types: [[FARGS_PTR]])
  *)
  (* Let's create the LLVM-IR function now. *)
  let name = "tfun" in
  let fty =
    Llvm.function_type (Llvm.void_type context) [| Llvm.i32_type context |]
  in
  let f = Llvm.define_function name fty m in
  let f_di =
    Llvm_debuginfo.dibuild_create_function dibuilder ~scope:m_di ~name
      ~linkage_name:name ~file:file_di ~line_no:10 ~ty:fty_di
      ~is_local_to_unit:false ~is_definition:true ~scope_line:10
      ~flags:flags_zero ~is_optimized:false
  in
  stdout_metadata f_di;
  (* CHECK: [[SBPRG_PTR:<0x[0-9a-f]*>]] = distinct !DISubprogram(name: "tfun", linkageName: "tfun", scope: [[MODULE_PTR]], file: [[FILE_PTR]], line: 10, type: [[SBRTNTY_PTR]], scopeLine: 10, spFlags: DISPFlagDefinition, unit: [[CMPUNIT_PTR]], retainedNodes: {{<0x[0-9a-f]*>}})
  *)
  Llvm_debuginfo.set_subprogram f f_di;
  ( match Llvm_debuginfo.get_subprogram f with
  | Some f_di' -> insist (f_di = f_di')
  | None -> insist false );
  insist
    ( Llvm_debuginfo.get_metadata_kind f_di
    = Llvm_debuginfo.MetadataKind.DISubprogramMetadataKind );
  insist (Llvm_debuginfo.di_subprogram_get_line f_di = 10);
  (fty, f, f_di)

let test_bbinstr fty f f_di file_di dibuilder =
  group "basic_block and instructions tests";
  (* Create this pattern:
   *   if (arg0 != 0) {
   *      foo(arg0);
   *   }
   *   return;
   *)
  let arg0 = (Llvm.params f).(0) in
  let builder = Llvm.builder_at_end context (Llvm.entry_block f) in
  let zero = Llvm.const_int (Llvm.i32_type context) 0 in
  let cmpi = Llvm.build_icmp Llvm.Icmp.Ne zero arg0 "cmpi" builder in
  let truebb = Llvm.append_block context "truebb" f in
  let falsebb = Llvm.append_block context "falsebb" f in
  let _ = Llvm.build_cond_br cmpi truebb falsebb builder in
  let foodecl = Llvm.declare_function "foo" fty (Llvm.global_parent f) in
  let _ =
    Llvm.position_at_end truebb builder;
    let scope =
      Llvm_debuginfo.dibuild_create_lexical_block dibuilder ~scope:f_di
        ~file:file_di ~line:9 ~column:4
    in
    let file_of_f_di = Llvm_debuginfo.di_scope_get_file ~scope:f_di in
    let file_of_scope = Llvm_debuginfo.di_scope_get_file ~scope in
    insist
      ( match (file_of_f_di, file_of_scope) with
      | Some file_of_f_di', Some file_of_scope' ->
          file_of_f_di' = file_di && file_of_scope' = file_di
      | _ -> false );
    let foocall = Llvm.build_call2 fty foodecl [| arg0 |] "" builder in
    let foocall_loc =
      Llvm_debuginfo.dibuild_create_debug_location context ~line:10 ~column:12
        ~scope
    in
    Llvm_debuginfo.instr_set_debug_loc foocall (Some foocall_loc);
    insist
      ( match Llvm_debuginfo.instr_get_debug_loc foocall with
      | Some foocall_loc' -> foocall_loc' = foocall_loc
      | None -> false );
    stdout_metadata scope;
    (* CHECK: [[BLOCK_PTR:<0x[0-9a-f]*>]] = distinct !DILexicalBlock(scope: [[SBPRG_PTR]], file: [[FILE_PTR]], line: 9, column: 4)
     *)
    stdout_metadata foocall_loc;
    (* CHECK: !DILocation(line: 10, column: 12, scope: [[BLOCK_PTR]])
     *)
    insist
      ( Llvm_debuginfo.di_location_get_scope ~location:foocall_loc = scope
      && Llvm_debuginfo.di_location_get_line ~location:foocall_loc = 10
      && Llvm_debuginfo.di_location_get_column ~location:foocall_loc = 12 );
    insist
      ( Llvm_debuginfo.get_metadata_kind foocall_loc
        = Llvm_debuginfo.MetadataKind.DILocationMetadataKind
      && Llvm_debuginfo.get_metadata_kind scope
         = Llvm_debuginfo.MetadataKind.DILexicalBlockMetadataKind );
    Llvm.build_br falsebb builder
  in
  let _ =
    Llvm.position_at_end falsebb builder;
    Llvm.build_ret_void builder
  in
  (* Printf.printf "%s\n" (Llvm.string_of_llmodule (Llvm.global_parent f)); *)
  ()

let test_global_variable_expression dibuilder f_di m_di =
  group "global variable expression tests";
  let cexpr_di =
    Llvm_debuginfo.dibuild_create_constant_value_expression dibuilder 0
  in
  stdout_metadata cexpr_di;
  (* CHECK: [[DICEXPR:!DIExpression\(DW_OP_constu, 0, DW_OP_stack_value\)]]
   *)
  insist
    ( Llvm_debuginfo.get_metadata_kind cexpr_di
    = Llvm_debuginfo.MetadataKind.DIExpressionMetadataKind );
  let ty = int_ty_di 64 dibuilder in
  stdout_metadata ty;
  (* CHECK: [[INT64TY_PTR:<0x[0-9a-f]*>]] = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_signed)
   *)
  let gvexpr_di =
    Llvm_debuginfo.dibuild_create_global_variable_expression dibuilder
      ~scope:m_di ~name:"my_global" ~linkage:"" ~file:f_di ~line:5 ~ty
      ~is_local_to_unit:true ~expr:cexpr_di ~decl:null_metadata ~align_in_bits:0
  in
  insist
    ( Llvm_debuginfo.get_metadata_kind gvexpr_di
    = Llvm_debuginfo.MetadataKind.DIGlobalVariableExpressionMetadataKind );
  ( match
      Llvm_debuginfo.di_global_variable_expression_get_variable gvexpr_di
    with
  | Some gvexpr_var_di ->
      insist
        ( Llvm_debuginfo.get_metadata_kind gvexpr_var_di
        = Llvm_debuginfo.MetadataKind.DIGlobalVariableMetadataKind );
      stdout_metadata gvexpr_var_di
      (* CHECK: [[GV_PTR:<0x[0-9a-f]*>]] = distinct !DIGlobalVariable(name: "my_global", scope: [[MODULE_PTR]], file: [[FILE_PTR]], line: 5, type: [[INT64TY_PTR]], isLocal: true, isDefinition: true)
       *)
  | None -> insist false );
  stdout_metadata gvexpr_di;
  (* CHECK: [[GVEXP_PTR:<0x[0-9a-f]*>]] = !DIGlobalVariableExpression(var: [[GV_PTR]], expr: [[DICEXPR]])
   *)
  ()

let test_variables f dibuilder file_di fun_di =
  let entry_term = Option.get @@ (Llvm.block_terminator (Llvm.entry_block f)) in
  group "Local and parameter variable tests";
  let ty = int_ty_di 64 dibuilder in
  stdout_metadata ty;
  (* CHECK: [[INT64TY_PTR:<0x[0-9a-f]*>]] = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_signed)
  *)
  let auto_var =
    Llvm_debuginfo.dibuild_create_auto_variable dibuilder ~scope:fun_di
      ~name:"my_local" ~file:file_di ~line:10 ~ty
      ~always_preserve:false flags_zero ~align_in_bits:0
  in
  stdout_metadata auto_var;
  (* CHECK: [[LOCAL_VAR_PTR:<0x[0-9a-f]*>]] = !DILocalVariable(name: "my_local", scope: <{{0x[0-9a-f]*}}>, file: <{{0x[0-9a-f]*}}>, line: 10, type: [[INT64TY_PTR]])
  *)
  let builder = Llvm.builder_before context entry_term in
  let all = Llvm.build_alloca (Llvm.i64_type context)  "my_alloca" builder in
  let scope =
    Llvm_debuginfo.dibuild_create_lexical_block dibuilder ~scope:fun_di
      ~file:file_di ~line:9 ~column:4
  in
  let location =
    Llvm_debuginfo.dibuild_create_debug_location
    context ~line:10 ~column:12 ~scope
  in
  let vdi = Llvm_debuginfo.dibuild_insert_declare_before dibuilder ~storage:all
    ~var_info:auto_var ~expr:(Llvm_debuginfo.dibuild_expression dibuilder [||])
    ~location ~instr:entry_term
  in
  let () = Printf.printf "%s\n" (Llvm.string_of_llvalue vdi) in
  (* CHECK: call void @llvm.dbg.declare(metadata ptr %my_alloca, metadata {{![0-9]+}}, metadata !DIExpression()), !dbg {{\![0-9]+}}
  *)
  let arg0 = (Llvm.params f).(0) in
  let arg_var = Llvm_debuginfo.dibuild_create_parameter_variable dibuilder ~scope:fun_di
    ~name:"my_arg" ~argno:0 ~file:file_di ~line:10 ~ty
    ~always_preserve:false flags_zero
  in
  let argdi = Llvm_debuginfo.dibuild_insert_declare_before dibuilder ~storage:arg0
    ~var_info:arg_var ~expr:(Llvm_debuginfo.dibuild_expression dibuilder [||])
    ~location ~instr:entry_term
  in
  let () = Printf.printf "%s\n" (Llvm.string_of_llvalue argdi) in
  (* CHECK: call void @llvm.dbg.declare(metadata i32 %0, metadata {{![0-9]+}}, metadata !DIExpression()), !dbg {{\![0-9]+}}
  *)
  ()

let test_types dibuilder file_di m_di =
  group "type tests";
  let namespace_di =
    Llvm_debuginfo.dibuild_create_namespace dibuilder ~parent_ref:m_di
      ~name:"NameSpace1" ~export_symbols:false
  in
  stdout_metadata namespace_di;
  (* CHECK: [[NAMESPACE_PTR:<0x[0-9a-f]*>]] = !DINamespace(name: "NameSpace1", scope: [[MODULE_PTR]])
   *)
  let int64_ty_di = int_ty_di 64 dibuilder in
  let structty_args = [| int64_ty_di; int64_ty_di; int64_ty_di |] in
  let struct_ty_di =
    Llvm_debuginfo.dibuild_create_struct_type dibuilder ~scope:namespace_di
      ~name:"StructType1" ~file:file_di ~line_number:20 ~size_in_bits:192
      ~align_in_bits:0 flags_zero ~derived_from:null_metadata
      ~elements:structty_args Llvm_debuginfo.DWARFSourceLanguageKind.C89
      ~vtable_holder:null_metadata ~unique_id:"StructType1"
  in
  (* Since there's no way to fetch the element types which is now
   * a type array, we build that again for checking. *)
  let structty_di_eltypes =
    Llvm_debuginfo.dibuild_get_or_create_type_array dibuilder
      ~data:structty_args
  in
  stdout_metadata structty_di_eltypes;
  (* CHECK: [[STRUCTELT_PTR:<0x[0-9a-f]*>]] = !{[[INT64TY_PTR]], [[INT64TY_PTR]], [[INT64TY_PTR]]}
   *)
  stdout_metadata struct_ty_di;
  (* CHECK: [[STRUCT_PTR:<0x[0-9a-f]*>]] = !DICompositeType(tag: DW_TAG_structure_type, name: "StructType1", scope: [[NAMESPACE_PTR]], file: [[FILE_PTR]], line: 20, size: 192, elements: [[STRUCTELT_PTR]], identifier: "StructType1")
   *)
  insist
    ( Llvm_debuginfo.get_metadata_kind struct_ty_di
    = Llvm_debuginfo.MetadataKind.DICompositeTypeMetadataKind );
  let structptr_di =
    Llvm_debuginfo.dibuild_create_pointer_type dibuilder
      ~pointee_ty:struct_ty_di ~size_in_bits:192 ~align_in_bits:0
      ~address_space:0 ~name:""
  in
  stdout_metadata structptr_di;
  (* CHECK: [[STRUCTPTR_PTR:<0x[0-9a-f]*>]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: [[STRUCT_PTR]], size: 192, dwarfAddressSpace: 0)
   *)
  insist
    ( Llvm_debuginfo.get_metadata_kind structptr_di
    = Llvm_debuginfo.MetadataKind.DIDerivedTypeMetadataKind );
  let enumerator1 =
    Llvm_debuginfo.dibuild_create_enumerator dibuilder ~name:"Test_A" ~value:0
      ~is_unsigned:true
  in
  stdout_metadata enumerator1;
  (* CHECK: [[ENUMERATOR1_PTR:<0x[0-9a-f]*>]] = !DIEnumerator(name: "Test_A", value: 0, isUnsigned: true)
   *)
  let enumerator2 =
    Llvm_debuginfo.dibuild_create_enumerator dibuilder ~name:"Test_B" ~value:1
      ~is_unsigned:true
  in
  stdout_metadata enumerator2;
  (* CHECK: [[ENUMERATOR2_PTR:<0x[0-9a-f]*>]] = !DIEnumerator(name: "Test_B", value: 1, isUnsigned: true)
   *)
  let enumerator3 =
    Llvm_debuginfo.dibuild_create_enumerator dibuilder ~name:"Test_C" ~value:2
      ~is_unsigned:true
  in
  insist
    ( Llvm_debuginfo.get_metadata_kind enumerator1
      = Llvm_debuginfo.MetadataKind.DIEnumeratorMetadataKind
    && Llvm_debuginfo.get_metadata_kind enumerator2
       = Llvm_debuginfo.MetadataKind.DIEnumeratorMetadataKind
    && Llvm_debuginfo.get_metadata_kind enumerator3
       = Llvm_debuginfo.MetadataKind.DIEnumeratorMetadataKind );
  stdout_metadata enumerator3;
  (* CHECK: [[ENUMERATOR3_PTR:<0x[0-9a-f]*>]] = !DIEnumerator(name: "Test_C", value: 2, isUnsigned: true)
   *)
  let elements = [| enumerator1; enumerator2; enumerator3 |] in
  let enumeration_ty_di =
    Llvm_debuginfo.dibuild_create_enumeration_type dibuilder ~scope:namespace_di
      ~name:"EnumTest" ~file:file_di ~line_number:1 ~size_in_bits:64
      ~align_in_bits:0 ~elements ~class_ty:int64_ty_di
  in
  let elements_arr =
    Llvm_debuginfo.dibuild_get_or_create_array dibuilder ~data:elements
  in
  stdout_metadata elements_arr;
  (* CHECK: [[ELEMENTS_PTR:<0x[0-9a-f]*>]] = !{[[ENUMERATOR1_PTR]], [[ENUMERATOR2_PTR]], [[ENUMERATOR3_PTR]]}
   *)
  stdout_metadata enumeration_ty_di;
  (* CHECK: [[ENUMERATION_PTR:<0x[0-9a-f]*>]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "EnumTest", scope: [[NAMESPACE_PTR]], file: [[FILE_PTR]], line: 1, baseType: [[INT64TY_PTR]], size: 64, elements: [[ELEMENTS_PTR]])
   *)
  insist
    ( Llvm_debuginfo.get_metadata_kind enumeration_ty_di
    = Llvm_debuginfo.MetadataKind.DICompositeTypeMetadataKind );
  let int32_ty_di = int_ty_di 32 dibuilder in
  let class_mem1 =
    Llvm_debuginfo.dibuild_create_member_type dibuilder ~scope:namespace_di
      ~name:"Field1" ~file:file_di ~line_number:3 ~size_in_bits:32
      ~align_in_bits:0 ~offset_in_bits:0 flags_zero ~ty:int32_ty_di
  in
  stdout_metadata class_mem1;
  (* CHECK: [[MEMB1_PTR:<0x[0-9a-f]*>]] = !DIDerivedType(tag: DW_TAG_member, name: "Field1", scope: [[NAMESPACE_PTR]], file: [[FILE_PTR]], line: 3, baseType: [[INT32_PTR]], size: 32)
   *)
  insist (Llvm_debuginfo.di_type_get_name class_mem1 = "Field1");
  insist (Llvm_debuginfo.di_type_get_line class_mem1 = 3);
  let class_mem2 =
    Llvm_debuginfo.dibuild_create_member_type dibuilder ~scope:namespace_di
      ~name:"Field2" ~file:file_di ~line_number:4 ~size_in_bits:64
      ~align_in_bits:8 ~offset_in_bits:32 flags_zero ~ty:int64_ty_di
  in
  stdout_metadata class_mem2;
  (* CHECK: [[MEMB2_PTR:<0x[0-9a-f]*>]] = !DIDerivedType(tag: DW_TAG_member, name: "Field2", scope: [[NAMESPACE_PTR]], file: [[FILE_PTR]], line: 4, baseType: [[INT64TY_PTR]], size: 64, align: 8, offset: 32)
   *)
  insist (Llvm_debuginfo.di_type_get_offset_in_bits class_mem2 = 32);
  insist (Llvm_debuginfo.di_type_get_size_in_bits class_mem2 = 64);
  insist (Llvm_debuginfo.di_type_get_align_in_bits class_mem2 = 8);
  let class_elements = [| class_mem1; class_mem2 |] in
  insist
    ( Llvm_debuginfo.get_metadata_kind class_mem1
      = Llvm_debuginfo.MetadataKind.DIDerivedTypeMetadataKind
    && Llvm_debuginfo.get_metadata_kind class_mem2
       = Llvm_debuginfo.MetadataKind.DIDerivedTypeMetadataKind );
  stdout_metadata
    (Llvm_debuginfo.dibuild_get_or_create_type_array dibuilder
       ~data:class_elements);
  (* CHECK: [[CLASSMEM_PTRS:<0x[0-9a-f]*>]] = !{[[MEMB1_PTR]], [[MEMB2_PTR]]}
   *)
  let classty_di =
    Llvm_debuginfo.dibuild_create_class_type dibuilder ~scope:namespace_di
      ~name:"MyClass" ~file:file_di ~line_number:1 ~size_in_bits:96
      ~align_in_bits:0 ~offset_in_bits:0 flags_zero ~derived_from:null_metadata
      ~elements:class_elements ~vtable_holder:null_metadata
      ~template_params_node:null_metadata ~unique_identifier:"MyClass"
  in
  stdout_metadata classty_di;
  (* [[CLASS_PTR:<0x[0-9a-f]*>]] = !DICompositeType(tag: DW_TAG_structure_type, name: "MyClass", scope: [[NAMESPACE_PTR]], file: [[FILE_PTR]], line: 1, size: 96, elements: [[CLASSMEM_PTRS]], identifier: "MyClass")
   *)
  insist
    ( Llvm_debuginfo.get_metadata_kind classty_di
    = Llvm_debuginfo.MetadataKind.DICompositeTypeMetadataKind );
  ()

let () =
  let m, dibuilder, file_di, m_di = test_get_module () in
  let fty, f, fun_di = test_get_function m dibuilder file_di m_di in
  let () = test_bbinstr fty f fun_di file_di dibuilder in
  let () = test_global_variable_expression dibuilder file_di m_di in
  let () = test_variables f dibuilder file_di fun_di in
  let () = test_types dibuilder file_di m_di in
  Llvm_debuginfo.dibuild_finalize dibuilder;
  ( match Llvm_analysis.verify_module m with
  | Some err ->
      prerr_endline ("Verification of module failed: " ^ err);
      exit_status := 1
  | None -> () );
  exit !exit_status
