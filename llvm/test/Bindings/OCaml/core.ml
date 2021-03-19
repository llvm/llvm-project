(* RUN: rm -rf %t && mkdir -p %t && cp %s %t/core.ml
 * RUN: %ocamlc -g -w +A -package llvm.analysis -package llvm.bitwriter -linkpkg %t/core.ml -o %t/executable
 * RUN: %t/executable %t/bitcode.bc
 * RUN: %ocamlopt -g -w +A -package llvm.analysis -package llvm.bitwriter -linkpkg %t/core.ml -o %t/executable
 * RUN: %t/executable %t/bitcode.bc
 * RUN: llvm-dis < %t/bitcode.bc > %t/dis.ll
 * RUN: FileCheck %s < %t/dis.ll
 * Do a second pass for things that shouldn't be anywhere.
 * RUN: FileCheck -check-prefix=CHECK-NOWHERE %s < %t/dis.ll
 * XFAIL: vg_leak
 *)

(* Note: It takes several seconds for ocamlopt to link an executable with
         libLLVMCore.a, so it's better to write a big test than a bunch of
         little ones. *)

open Llvm
open Llvm_bitwriter


(* Tiny unit test framework - really just to help find which line is busted *)
let exit_status = ref 0
let suite_name = ref ""
let group_name = ref ""
let case_num = ref 0
let print_checkpoints = false
let context = global_context ()
let i1_type = Llvm.i1_type context
let i8_type = Llvm.i8_type context
let i16_type = Llvm.i16_type context
let i32_type = Llvm.i32_type context
let i64_type = Llvm.i64_type context
let void_type = Llvm.void_type context
let float_type = Llvm.float_type context
let double_type = Llvm.double_type context
let fp128_type = Llvm.fp128_type context

let group name =
  group_name := !suite_name ^ "/" ^ name;
  case_num := 0;
  if print_checkpoints then
    prerr_endline ("  " ^ name ^ "...")

let insist cond =
  incr case_num;
  if not cond then
    exit_status := 10;
  match print_checkpoints, cond with
  | false, true -> ()
  | false, false ->
      prerr_endline ("FAILED: " ^ !suite_name ^ "/" ^ !group_name ^ " #" ^ (string_of_int !case_num))
  | true, true ->
      prerr_endline ("    " ^ (string_of_int !case_num))
  | true, false ->
      prerr_endline ("    " ^ (string_of_int !case_num) ^ " FAIL")

let suite name f =
  suite_name := name;
  if print_checkpoints then
    prerr_endline (name ^ ":");
  f ()


(*===-- Fixture -----------------------------------------------------------===*)

let filename = Sys.argv.(1)
let m = create_module context filename

(*===-- Contained types  --------------------------------------------------===*)

let test_contained_types () =
  let pointer_i32 = pointer_type i32_type in
  insist (i32_type = (Array.get (subtypes pointer_i32) 0));

  let ar = struct_type context [| i32_type; i8_type |] in
  insist (i32_type = (Array.get (subtypes ar)) 0);
  insist (i8_type = (Array.get (subtypes ar)) 1)


(*===-- Conversion --------------------------------------------------------===*)

let test_conversion () =
  insist ("i32" = (string_of_lltype i32_type));
  let c = const_int i32_type 42 in
  insist ("i32 42" = (string_of_llvalue c))


(*===-- Target ------------------------------------------------------------===*)

let test_target () =
  begin group "triple";
    let trip = "i686-apple-darwin8" in
    set_target_triple trip m;
    insist (trip = target_triple m)
  end;

  begin group "layout";
    let layout = "e" in
    set_data_layout layout m;
    insist (layout = data_layout m)
  end
  (* CHECK: target datalayout = "e"
   * CHECK: target triple = "i686-apple-darwin8"
   *)


(*===-- Constants ---------------------------------------------------------===*)

let test_constants () =
  (* CHECK: const_int{{.*}}i32{{.*}}-1
   *)
  group "int";
  let c = const_int i32_type (-1) in
  ignore (define_global "const_int" c m);
  insist (i32_type = type_of c);
  insist (is_constant c);
  insist (Some (-1L) = int64_of_const c);

  (* CHECK: const_sext_int{{.*}}i64{{.*}}-1
   *)
  group "sext int";
  let c = const_int i64_type (-1) in
  ignore (define_global "const_sext_int" c m);
  insist (i64_type = type_of c);
  insist (Some (-1L) = int64_of_const c);

  (* CHECK: const_zext_int64{{.*}}i64{{.*}}4294967295
   *)
  group "zext int64";
  let c = const_of_int64 i64_type (Int64.of_string "4294967295") false in
  ignore (define_global "const_zext_int64" c m);
  insist (i64_type = type_of c);
  insist (Some 4294967295L = int64_of_const c);

  (* CHECK: const_int_string{{.*}}i32{{.*}}-1
   *)
  group "int string";
  let c = const_int_of_string i32_type "-1" 10 in
  ignore (define_global "const_int_string" c m);
  insist (i32_type = type_of c);
  insist (None = (string_of_const c));
  insist (None = float_of_const c);
  insist (Some (-1L) = int64_of_const c);

  (* CHECK: const_int64{{.*}}i64{{.*}}9223372036854775807
   *)
  group "max int64";
  let c = const_of_int64 i64_type 9223372036854775807L true in
  ignore (define_global "const_int64" c m) ;
  insist (i64_type = type_of c);
  insist (Some 9223372036854775807L = int64_of_const c);

  if Sys.word_size = 64; then begin
    group "long int";
    let c = const_int i64_type (1 lsl 61) in
    insist (c = const_of_int64 i64_type (Int64.of_int (1 lsl 61)) false)
  end;

  (* CHECK: @const_string = global {{.*}}c"cruel\00world"
   *)
  group "string";
  let c = const_string context "cruel\000world" in
  ignore (define_global "const_string" c m);
  insist ((array_type i8_type 11) = type_of c);
  insist ((Some "cruel\000world") = (string_of_const c));

  (* CHECK: const_stringz{{.*}}"hi\00again\00"
   *)
  group "stringz";
  let c = const_stringz context "hi\000again" in
  ignore (define_global "const_stringz" c m);
  insist ((array_type i8_type 9) = type_of c);

  (* CHECK: const_single{{.*}}2.75
   * CHECK: const_double{{.*}}3.1459
   * CHECK: const_double_string{{.*}}2
   * CHECK: const_fake_fp128{{.*}}0xL00000000000000004000000000000000
   * CHECK: const_fp128_string{{.*}}0xLF3CB1CCF26FBC178452FB4EC7F91973F
   *)
  begin group "real";
    let cs = const_float float_type 2.75 in
    ignore (define_global "const_single" cs m);
    insist (float_type = type_of cs);
    insist (float_of_const cs = Some 2.75);

    let cd = const_float double_type 3.1459 in
    ignore (define_global "const_double" cd m);
    insist (double_type = type_of cd);
    insist (float_of_const cd = Some 3.1459);

    let cd = const_float_of_string double_type "2" in
    ignore (define_global "const_double_string" cd m);
    insist (double_type = type_of cd);
    insist (float_of_const cd = Some 2.);

    let cd = const_float fp128_type 2. in
    ignore (define_global "const_fake_fp128" cd m);
    insist (fp128_type = type_of cd);
    insist (float_of_const cd = Some 2.);

    let cd = const_float_of_string fp128_type "1e400" in
    ignore (define_global "const_fp128_string" cd m);
    insist (fp128_type = type_of cd);
    insist (float_of_const cd = None);
  end;

  let one = const_int i16_type 1 in
  let two = const_int i16_type 2 in
  let three = const_int i32_type 3 in
  let four = const_int i32_type 4 in

  (* CHECK: const_array{{.*}}[i32 3, i32 4]
   *)
  group "array";
  let c = const_array i32_type [| three; four |] in
  ignore (define_global "const_array" c m);
  insist ((array_type i32_type 2) = (type_of c));
  insist (three = (const_element c 0));
  insist (four = (const_element c 1));

  (* CHECK: const_vector{{.*}}<i16 1, i16 2{{.*}}>
   *)
  group "vector";
  let c = const_vector [| one; two; one; two;
                          one; two; one; two |] in
  ignore (define_global "const_vector" c m);
  insist ((vector_type i16_type 8) = (type_of c));

  (* CHECK: const_structure{{.*.}}i16 1, i16 2, i32 3, i32 4
   *)
  group "structure";
  let c = const_struct context [| one; two; three; four |] in
  ignore (define_global "const_structure" c m);
  insist ((struct_type context [| i16_type; i16_type; i32_type; i32_type |])
        = (type_of c));

  (* CHECK: const_null{{.*}}zeroinit
   *)
  group "null";
  let c = const_null (packed_struct_type context [| i1_type; i8_type; i64_type;
                                                    double_type |]) in
  ignore (define_global "const_null" c m);

  (* CHECK: const_all_ones{{.*}}-1
   *)
  group "all ones";
  let c = const_all_ones i64_type in
  ignore (define_global "const_all_ones" c m);

  group "pointer null"; begin
    (* CHECK: const_pointer_null = global i64* null
     *)
    let c = const_pointer_null (pointer_type i64_type) in
    ignore (define_global "const_pointer_null" c m);
  end;

  (* CHECK: const_undef{{.*}}undef
   *)
  group "undef";
  let c = undef i1_type in
  ignore (define_global "const_undef" c m);
  insist (i1_type = type_of c);
  insist (is_undef c);

  (* CHECK: const_poison{{.*}}poison
   *)
  group "poison";
  let c = poison i1_type in
  ignore (define_global "const_poison" c m);
  insist (i1_type = type_of c);
  insist (is_poison c);

  group "constant arithmetic";
  (* CHECK: @const_neg = global i64 sub
   * CHECK: @const_nsw_neg = global i64 sub nsw
   * CHECK: @const_nuw_neg = global i64 sub nuw
   * CHECK: @const_fneg = global double fneg
   * CHECK: @const_not = global i64 xor
   * CHECK: @const_add = global i64 add
   * CHECK: @const_nsw_add = global i64 add nsw
   * CHECK: @const_nuw_add = global i64 add nuw
   * CHECK: @const_fadd = global double fadd
   * CHECK: @const_sub = global i64 sub
   * CHECK: @const_nsw_sub = global i64 sub nsw
   * CHECK: @const_nuw_sub = global i64 sub nuw
   * CHECK: @const_fsub = global double fsub
   * CHECK: @const_mul = global i64 mul
   * CHECK: @const_nsw_mul = global i64 mul nsw
   * CHECK: @const_nuw_mul = global i64 mul nuw
   * CHECK: @const_fmul = global double fmul
   * CHECK: @const_udiv = global i64 udiv
   * CHECK: @const_sdiv = global i64 sdiv
   * CHECK: @const_exact_sdiv = global i64 sdiv exact
   * CHECK: @const_fdiv = global double fdiv
   * CHECK: @const_urem = global i64 urem
   * CHECK: @const_srem = global i64 srem
   * CHECK: @const_frem = global double frem
   * CHECK: @const_and = global i64 and
   * CHECK: @const_or = global i64 or
   * CHECK: @const_xor = global i64 xor
   * CHECK: @const_icmp = global i1 icmp sle
   * CHECK: @const_fcmp = global i1 fcmp ole
   *)
  let void_ptr = pointer_type i8_type in
  let five = const_int i64_type 5 in
  let ffive = const_uitofp five double_type in
  let foldbomb_gv = define_global "FoldBomb" (const_null i8_type) m in
  let foldbomb = const_ptrtoint foldbomb_gv i64_type in
  let ffoldbomb = const_uitofp foldbomb double_type in
  ignore (define_global "const_neg" (const_neg foldbomb) m);
  ignore (define_global "const_nsw_neg" (const_nsw_neg foldbomb) m);
  ignore (define_global "const_nuw_neg" (const_nuw_neg foldbomb) m);
  ignore (define_global "const_fneg" (const_fneg ffoldbomb) m);
  ignore (define_global "const_not" (const_not foldbomb) m);
  ignore (define_global "const_add" (const_add foldbomb five) m);
  ignore (define_global "const_nsw_add" (const_nsw_add foldbomb five) m);
  ignore (define_global "const_nuw_add" (const_nuw_add foldbomb five) m);
  ignore (define_global "const_fadd" (const_fadd ffoldbomb ffive) m);
  ignore (define_global "const_sub" (const_sub foldbomb five) m);
  ignore (define_global "const_nsw_sub" (const_nsw_sub foldbomb five) m);
  ignore (define_global "const_nuw_sub" (const_nuw_sub foldbomb five) m);
  ignore (define_global "const_fsub" (const_fsub ffoldbomb ffive) m);
  ignore (define_global "const_mul" (const_mul foldbomb five) m);
  ignore (define_global "const_nsw_mul" (const_nsw_mul foldbomb five) m);
  ignore (define_global "const_nuw_mul" (const_nuw_mul foldbomb five) m);
  ignore (define_global "const_fmul" (const_fmul ffoldbomb ffive) m);
  ignore (define_global "const_udiv" (const_udiv foldbomb five) m);
  ignore (define_global "const_sdiv" (const_sdiv foldbomb five) m);
  ignore (define_global "const_exact_sdiv" (const_exact_sdiv foldbomb five) m);
  ignore (define_global "const_fdiv" (const_fdiv ffoldbomb ffive) m);
  ignore (define_global "const_urem" (const_urem foldbomb five) m);
  ignore (define_global "const_srem" (const_srem foldbomb five) m);
  ignore (define_global "const_frem" (const_frem ffoldbomb ffive) m);
  ignore (define_global "const_and" (const_and foldbomb five) m);
  ignore (define_global "const_or" (const_or foldbomb five) m);
  ignore (define_global "const_xor" (const_xor foldbomb five) m);
  ignore (define_global "const_icmp" (const_icmp Icmp.Sle foldbomb five) m);
  ignore (define_global "const_fcmp" (const_fcmp Fcmp.Ole ffoldbomb ffive) m);

  group "constant casts";
  (* CHECK: const_trunc{{.*}}trunc
   * CHECK: const_sext{{.*}}sext
   * CHECK: const_zext{{.*}}zext
   * CHECK: const_fptrunc{{.*}}fptrunc
   * CHECK: const_fpext{{.*}}fpext
   * CHECK: const_uitofp{{.*}}uitofp
   * CHECK: const_sitofp{{.*}}sitofp
   * CHECK: const_fptoui{{.*}}fptoui
   * CHECK: const_fptosi{{.*}}fptosi
   * CHECK: const_ptrtoint{{.*}}ptrtoint
   * CHECK: const_inttoptr{{.*}}inttoptr
   * CHECK: const_bitcast{{.*}}bitcast
   * CHECK: const_intcast{{.*}}zext
   *)
  let i128_type = integer_type context 128 in
  ignore (define_global "const_trunc" (const_trunc (const_add foldbomb five)
                                               i8_type) m);
  ignore (define_global "const_sext" (const_sext foldbomb i128_type) m);
  ignore (define_global "const_zext" (const_zext foldbomb i128_type) m);
  ignore (define_global "const_fptrunc" (const_fptrunc ffoldbomb float_type) m);
  ignore (define_global "const_fpext" (const_fpext ffoldbomb fp128_type) m);
  ignore (define_global "const_uitofp" (const_uitofp foldbomb double_type) m);
  ignore (define_global "const_sitofp" (const_sitofp foldbomb double_type) m);
  ignore (define_global "const_fptoui" (const_fptoui ffoldbomb i32_type) m);
  ignore (define_global "const_fptosi" (const_fptosi ffoldbomb i32_type) m);
  ignore (define_global "const_ptrtoint" (const_ptrtoint
    (const_gep (const_null (pointer_type i8_type))
               [| const_int i32_type 1 |])
    i32_type) m);
  ignore (define_global "const_inttoptr" (const_inttoptr (const_add foldbomb five)
                                                  void_ptr) m);
  ignore (define_global "const_bitcast" (const_bitcast ffoldbomb i64_type) m);
  ignore (define_global "const_intcast"
          (const_intcast foldbomb i128_type ~is_signed:false) m);

  group "misc constants";
  (* CHECK: const_size_of{{.*}}getelementptr{{.*}}null
   * CHECK: const_gep{{.*}}getelementptr
   * CHECK: const_select{{.*}}select
   * CHECK: const_extractelement{{.*}}extractelement
   * CHECK: const_insertelement{{.*}}insertelement
   * CHECK: const_shufflevector = global <4 x i32> <i32 0, i32 1, i32 1, i32 0>
   *)
  ignore (define_global "const_size_of" (size_of (pointer_type i8_type)) m);
  ignore (define_global "const_gep" (const_gep foldbomb_gv [| five |]) m);
  ignore (define_global "const_select" (const_select
    (const_icmp Icmp.Sle foldbomb five)
    (const_int i8_type (-1))
    (const_int i8_type 0)) m);
  let zero = const_int i32_type 0 in
  let one  = const_int i32_type 1 in
  ignore (define_global "const_extractelement" (const_extractelement
    (const_vector [| zero; one; zero; one |])
    (const_trunc foldbomb i32_type)) m);
  ignore (define_global "const_insertelement" (const_insertelement
    (const_vector [| zero; one; zero; one |])
    zero (const_trunc foldbomb i32_type)) m);
  ignore (define_global "const_shufflevector" (const_shufflevector
    (const_vector [| zero; one |])
    (const_vector [| one; zero |])
    (const_vector [| const_int i32_type 0; const_int i32_type 1;
                     const_int i32_type 2; const_int i32_type 3 |])) m);

  group "asm"; begin
    let ft = function_type void_type [| i32_type; i32_type; i32_type |] in
    ignore (const_inline_asm
      ft
      ""
      "{cx},{ax},{di},~{dirflag},~{fpsr},~{flags},~{edi},~{ecx}"
      true
      false)
  end;

  group "recursive struct"; begin
      let nsty = named_struct_type context "rec" in
      let pty = pointer_type nsty in
      struct_set_body nsty [| i32_type; pty |] false;
      let elts = [| const_int i32_type 4; const_pointer_null pty |] in
      let grec_init = const_named_struct nsty elts in
      ignore (define_global "grec" grec_init m);
      ignore (string_of_lltype nsty);
  end


(*===-- Attributes --------------------------------------------------------===*)

let test_attributes () =
  group "enum attrs";
  let nonnull_kind = enum_attr_kind "nonnull" in
  let dereferenceable_kind = enum_attr_kind "dereferenceable" in
  insist (nonnull_kind = (enum_attr_kind "nonnull"));
  insist (nonnull_kind <> dereferenceable_kind);

  let nonnull =
    create_enum_attr context "nonnull" 0L in
  let dereferenceable_4 =
    create_enum_attr context "dereferenceable" 4L in
  let dereferenceable_8 =
    create_enum_attr context "dereferenceable" 8L in
  insist (nonnull <> dereferenceable_4);
  insist (dereferenceable_4 <> dereferenceable_8);
  insist (nonnull = (create_enum_attr context "nonnull" 0L));
  insist ((repr_of_attr nonnull) =
          AttrRepr.Enum(nonnull_kind, 0L));
  insist ((repr_of_attr dereferenceable_4) =
          AttrRepr.Enum(dereferenceable_kind, 4L));
  insist ((attr_of_repr context (repr_of_attr nonnull)) =
          nonnull);
  insist ((attr_of_repr context (repr_of_attr dereferenceable_4)) =
          dereferenceable_4);

  group "string attrs";
  let foo_bar = create_string_attr context "foo" "bar" in
  let foo_baz = create_string_attr context "foo" "baz" in
  insist (foo_bar <> foo_baz);
  insist (foo_bar = (create_string_attr context "foo" "bar"));
  insist ((repr_of_attr foo_bar) = AttrRepr.String("foo", "bar"));
  insist ((attr_of_repr context (repr_of_attr foo_bar)) = foo_bar);
  ()

(*===-- Global Values -----------------------------------------------------===*)

let test_global_values () =
  let (++) x f = f x; x in
  let zero32 = const_null i32_type in

  (* CHECK: GVal01
   *)
  group "naming";
  let g = define_global "TEMPORARY" zero32 m in
  insist ("TEMPORARY" = value_name g);
  set_value_name "GVal01" g;
  insist ("GVal01" = value_name g);

  (* CHECK: GVal02{{.*}}linkonce
   *)
  group "linkage";
  let g = define_global "GVal02" zero32 m ++
          set_linkage Linkage.Link_once in
  insist (Linkage.Link_once = linkage g);

  (* CHECK: GVal03{{.*}}Hanalei
   *)
  group "section";
  let g = define_global "GVal03" zero32 m ++
          set_section "Hanalei" in
  insist ("Hanalei" = section g);

  (* CHECK: GVal04{{.*}}hidden
   *)
  group "visibility";
  let g = define_global "GVal04" zero32 m ++
          set_visibility Visibility.Hidden in
  insist (Visibility.Hidden = visibility g);

  (* CHECK: GVal05{{.*}}align 128
   *)
  group "alignment";
  let g = define_global "GVal05" zero32 m ++
          set_alignment 128 in
  insist (128 = alignment g);

  (* CHECK: GVal06{{.*}}dllexport
   *)
  group "dll_storage_class";
  let g = define_global "GVal06" zero32 m ++
          set_dll_storage_class DLLStorageClass.DLLExport in
  insist (DLLStorageClass.DLLExport = dll_storage_class g)


(*===-- Global Variables --------------------------------------------------===*)

let test_global_variables () =
  let (++) x f = f x; x in
  let forty_two32 = const_int i32_type 42 in

  group "declarations"; begin
    (* CHECK: @GVar01 = external global i32
     * CHECK: @QGVar01 = external addrspace(3) global i32
     *)
    insist (None == lookup_global "GVar01" m);
    let g = declare_global i32_type "GVar01" m in
    insist (is_declaration g);
    insist (pointer_type float_type ==
              type_of (declare_global float_type "GVar01" m));
    insist (g == declare_global i32_type "GVar01" m);
    insist (match lookup_global "GVar01" m with Some x -> x = g
                                              | None -> false);

    insist (None == lookup_global "QGVar01" m);
    let g = declare_qualified_global i32_type "QGVar01" 3 m in
    insist (is_declaration g);
    insist (qualified_pointer_type float_type 3 ==
              type_of (declare_qualified_global float_type "QGVar01" 3 m));
    insist (g == declare_qualified_global i32_type "QGVar01" 3 m);
    insist (match lookup_global "QGVar01" m with Some x -> x = g
                                              | None -> false);
  end;

  group "definitions"; begin
    (* CHECK: @GVar02 = global i32 42
     * CHECK: @GVar03 = global i32 42
     * CHECK: @QGVar02 = addrspace(3) global i32 42
     * CHECK: @QGVar03 = addrspace(3) global i32 42
     *)
    let g = define_global "GVar02" forty_two32 m in
    let g2 = declare_global i32_type "GVar03" m ++
           set_initializer forty_two32 in
    insist (not (is_declaration g));
    insist (not (is_declaration g2));
    insist ((global_initializer g) = (global_initializer g2));

    let g = define_qualified_global "QGVar02" forty_two32 3 m in
    let g2 = declare_qualified_global i32_type "QGVar03" 3 m ++
           set_initializer forty_two32 in
    insist (not (is_declaration g));
    insist (not (is_declaration g2));
    insist ((global_initializer g) = (global_initializer g2));
  end;

  (* CHECK: GVar04{{.*}}thread_local
   *)
  group "threadlocal";
  let g = define_global "GVar04" forty_two32 m ++
          set_thread_local true in
  insist (is_thread_local g);

  (* CHECK: GVar05{{.*}}thread_local(initialexec)
   *)
  group "threadlocal_mode";
  let g = define_global "GVar05" forty_two32 m ++
          set_thread_local_mode ThreadLocalMode.InitialExec in
  insist ((thread_local_mode g) = ThreadLocalMode.InitialExec);

  (* CHECK: GVar06{{.*}}externally_initialized
   *)
  group "externally_initialized";
  let g = define_global "GVar06" forty_two32 m ++
          set_externally_initialized true in
  insist (is_externally_initialized g);

  (* CHECK-NOWHERE-NOT: GVar07
   *)
  group "delete";
  let g = define_global "GVar07" forty_two32 m in
  delete_global g;

  (* CHECK: ConstGlobalVar{{.*}}constant
   *)
  group "constant";
  let g = define_global "ConstGlobalVar" forty_two32 m in
  insist (not (is_global_constant g));
  set_global_constant true g;
  insist (is_global_constant g);

  begin group "iteration";
    let m = create_module context "temp" in

    insist (get_module_identifier m = "temp");
    set_module_identifer m "temp2";
    insist (get_module_identifier m = "temp2");

    insist (At_end m = global_begin m);
    insist (At_start m = global_end m);

    let g1 = declare_global i32_type "One" m in
    let g2 = declare_global i32_type "Two" m in

    insist (Before g1 = global_begin m);
    insist (Before g2 = global_succ g1);
    insist (At_end m = global_succ g2);

    insist (After g2 = global_end m);
    insist (After g1 = global_pred g2);
    insist (At_start m = global_pred g1);

    let lf s x = s ^ "->" ^ value_name x in
    insist ("->One->Two" = fold_left_globals lf "" m);

    let rf x s = value_name x ^ "<-" ^ s in
    insist ("One<-Two<-" = fold_right_globals rf m "");

    dispose_module m
  end

(* String globals built below are emitted here.
 * CHECK: build_global_string{{.*}}stringval
 *)


(*===-- Uses --------------------------------------------------------------===*)

let test_uses () =
  let ty = function_type i32_type [| i32_type; i32_type |] in
  let fn = define_function "use_function" ty m in
  let b = builder_at_end context (entry_block fn) in

  let p1 = param fn 0 in
  let p2 = param fn 1 in
  let v1 = build_add p1 p2 "v1" b in
  let v2 = build_add p1 v1 "v2" b in
  let _ = build_add v1 v2 "v3" b in

  let lf s u = value_name (user u) ^ "->" ^ s in
  insist ("v2->v3->" = fold_left_uses lf "" v1);
  let rf u s = value_name (user u) ^ "<-" ^ s in
  insist ("v3<-v2<-" = fold_right_uses rf v1 "");

  let lf s u = value_name (used_value u) ^ "->" ^ s in
  insist ("v1->v1->" = fold_left_uses lf "" v1);

  let rf u s = value_name (used_value u) ^ "<-" ^ s in
  insist ("v1<-v1<-" = fold_right_uses rf v1 "");

  ignore (build_unreachable b)


(*===-- Users -------------------------------------------------------------===*)

let test_users () =
  let ty = function_type i32_type [| i32_type; i32_type |] in
  let fn = define_function "user_function" ty m in
  let b = builder_at_end context (entry_block fn) in

  let p1 = param fn 0 in
  let p2 = param fn 1 in
  let a3 = build_alloca i32_type "user_alloca" b in
  let p3 = build_load a3 "user_load" b in
  let i = build_add p1 p2 "sum" b in

  insist ((num_operands i) = 2);
  insist ((operand i 0) = p1);
  insist ((operand i 1) = p2);

  set_operand i 1 p3;
  insist ((operand i 1) != p2);
  insist ((operand i 1) = p3);

  ignore (build_unreachable b)


(*===-- Aliases -----------------------------------------------------------===*)

let test_aliases () =
  (* CHECK: @alias = alias i32, i32* @aliasee
   *)
  let forty_two32 = const_int i32_type 42 in
  let v = define_global "aliasee" forty_two32 m in
  ignore (add_alias m (pointer_type i32_type) v "alias")


(*===-- Functions ---------------------------------------------------------===*)

let test_functions () =
  let ty = function_type i32_type [| i32_type; i64_type |] in
  let ty2 = function_type i8_type [| i8_type; i64_type |] in

  (* CHECK: declare i32 @Fn1(i32, i64)
   *)
  begin group "declare";
    insist (None = lookup_function "Fn1" m);
    let fn = declare_function "Fn1" ty m in
    insist (pointer_type ty = type_of fn);
    insist (is_declaration fn);
    insist (0 = Array.length (basic_blocks fn));
    insist (pointer_type ty2 == type_of (declare_function "Fn1" ty2 m));
    insist (fn == declare_function "Fn1" ty m);
    insist (None <> lookup_function "Fn1" m);
    insist (match lookup_function "Fn1" m with Some x -> x = fn
                                             | None -> false);
    insist (m == global_parent fn)
  end;

  (* CHECK-NOWHERE-NOT: Fn2
   *)
  group "delete";
  let fn = declare_function "Fn2" ty m in
  delete_function fn;

  (* CHECK: define{{.*}}Fn3
   *)
  group "define";
  let fn = define_function "Fn3" ty m in
  insist (not (is_declaration fn));
  insist (1 = Array.length (basic_blocks fn));
  ignore (build_unreachable (builder_at_end context (entry_block fn)));

  (* CHECK: define{{.*}}Fn4{{.*}}Param1{{.*}}Param2
   *)
  group "params";
  let fn = define_function "Fn4" ty m in
  let params = params fn in
  insist (2 = Array.length params);
  insist (params.(0) = param fn 0);
  insist (params.(1) = param fn 1);
  insist (i32_type = type_of params.(0));
  insist (i64_type = type_of params.(1));
  set_value_name "Param1" params.(0);
  set_value_name "Param2" params.(1);
  ignore (build_unreachable (builder_at_end context (entry_block fn)));

  (* CHECK: fastcc{{.*}}Fn5
   *)
  group "callconv";
  let fn = define_function "Fn5" ty m in
  insist (CallConv.c = function_call_conv fn);
  set_function_call_conv CallConv.fast fn;
  insist (CallConv.fast = function_call_conv fn);
  ignore (build_unreachable (builder_at_end context (entry_block fn)));

  begin group "gc";
    (* CHECK: Fn6{{.*}}gc{{.*}}shadowstack
     *)
    let fn = define_function "Fn6" ty m in
    insist (None = gc fn);
    set_gc (Some "ocaml") fn;
    insist (Some "ocaml" = gc fn);
    set_gc None fn;
    insist (None = gc fn);
    set_gc (Some "shadowstack") fn;
    ignore (build_unreachable (builder_at_end context (entry_block fn)));
  end;

  begin group "iteration";
    let m = create_module context "temp" in

    insist (At_end m = function_begin m);
    insist (At_start m = function_end m);

    let f1 = define_function "One" ty m in
    let f2 = define_function "Two" ty m in

    insist (Before f1 = function_begin m);
    insist (Before f2 = function_succ f1);
    insist (At_end m = function_succ f2);

    insist (After f2 = function_end m);
    insist (After f1 = function_pred f2);
    insist (At_start m = function_pred f1);

    let lf s x = s ^ "->" ^ value_name x in
    insist ("->One->Two" = fold_left_functions lf "" m);

    let rf x s = value_name x ^ "<-" ^ s in
    insist ("One<-Two<-" = fold_right_functions rf m "");

    dispose_module m
  end


(*===-- Params ------------------------------------------------------------===*)

let test_params () =
  begin group "iteration";
    let m = create_module context "temp" in

    let vf = define_function "void" (function_type void_type [| |]) m in

    insist (At_end vf = param_begin vf);
    insist (At_start vf = param_end vf);

    let ty = function_type void_type [| i32_type; i32_type |] in
    let f = define_function "f" ty m in
    let p1 = param f 0 in
    let p2 = param f 1 in
    set_value_name "One" p1;
    set_value_name "Two" p2;

    insist (Before p1 = param_begin f);
    insist (Before p2 = param_succ p1);
    insist (At_end f = param_succ p2);

    insist (After p2 = param_end f);
    insist (After p1 = param_pred p2);
    insist (At_start f = param_pred p1);

    let lf s x = s ^ "->" ^ value_name x in
    insist ("->One->Two" = fold_left_params lf "" f);

    let rf x s = value_name x ^ "<-" ^ s in
    insist ("One<-Two<-" = fold_right_params rf f "");

    dispose_module m
  end


(*===-- Basic Blocks ------------------------------------------------------===*)

let test_basic_blocks () =
  let ty = function_type void_type [| |] in

  (* CHECK: Bb1
   *)
  group "entry";
  let fn = declare_function "X" ty m in
  let bb = append_block context "Bb1" fn in
  insist (bb = entry_block fn);
  ignore (build_unreachable (builder_at_end context bb));

  (* CHECK-NOWHERE-NOT: Bb2
   *)
  group "delete";
  let fn = declare_function "X2" ty m in
  let bb = append_block context "Bb2" fn in
  delete_block bb;

  group "insert";
  let fn = declare_function "X3" ty m in
  let bbb = append_block context "b" fn in
  let bba = insert_block context "a" bbb in
  insist ([| bba; bbb |] = basic_blocks fn);
  ignore (build_unreachable (builder_at_end context bba));
  ignore (build_unreachable (builder_at_end context bbb));

  (* CHECK: Bb3
   *)
  group "name/value";
  let fn = define_function "X4" ty m in
  let bb = entry_block fn in
  ignore (build_unreachable (builder_at_end context bb));
  let bbv = value_of_block bb in
  set_value_name "Bb3" bbv;
  insist ("Bb3" = value_name bbv);

  group "casts";
  let fn = define_function "X5" ty m in
  let bb = entry_block fn in
  ignore (build_unreachable (builder_at_end context bb));
  insist (bb = block_of_value (value_of_block bb));
  insist (value_is_block (value_of_block bb));
  insist (not (value_is_block (const_null i32_type)));

  begin group "iteration";
    let m = create_module context "temp" in
    let f = declare_function "Temp" (function_type i32_type [| |]) m in

    insist (At_end f = block_begin f);
    insist (At_start f = block_end f);

    let b1 = append_block context "One" f in
    let b2 = append_block context "Two" f in

    insist (Before b1 = block_begin f);
    insist (Before b2 = block_succ b1);
    insist (At_end f = block_succ b2);

    insist (After b2 = block_end f);
    insist (After b1 = block_pred b2);
    insist (At_start f = block_pred b1);

    let lf s x = s ^ "->" ^ value_name (value_of_block x) in
    insist ("->One->Two" = fold_left_blocks lf "" f);

    let rf x s = value_name (value_of_block x) ^ "<-" ^ s in
    insist ("One<-Two<-" = fold_right_blocks rf f "");

    dispose_module m
  end


(*===-- Instructions ------------------------------------------------------===*)

let test_instructions () =
  begin group "iteration";
    let m = create_module context "temp" in
    let fty = function_type void_type [| i32_type; i32_type |] in
    let f = define_function "f" fty m in
    let bb = entry_block f in
    let b = builder_at context (At_end bb) in

    insist (At_end bb = instr_begin bb);
    insist (At_start bb = instr_end bb);

    let i1 = build_add (param f 0) (param f 1) "One" b in
    let i2 = build_sub (param f 0) (param f 1) "Two" b in

    insist (Before i1 = instr_begin bb);
    insist (Before i2 = instr_succ i1);
    insist (At_end bb = instr_succ i2);

    insist (After i2 = instr_end bb);
    insist (After i1 = instr_pred i2);
    insist (At_start bb = instr_pred i1);

    let lf s x = s ^ "->" ^ value_name x in
    insist ("->One->Two" = fold_left_instrs lf "" bb);

    let rf x s = value_name x ^ "<-" ^ s in
    insist ("One<-Two<-" = fold_right_instrs rf bb "");

    dispose_module m
  end;

  group "clone instr";
  begin
    (* CHECK: %clone = add i32 %0, 2
     *)
    let fty = function_type void_type [| i32_type |] in
    let fn = define_function "BuilderParent" fty m in
    let bb = entry_block fn in
    let b = builder_at_end context bb in
    let p = param fn 0 in
    let sum = build_add p p "sum" b in
    let y = const_int i32_type 2 in
    let clone = instr_clone sum in
    set_operand clone 0 p;
    set_operand clone 1 y;
    insert_into_builder clone "clone" b;
    ignore (build_ret_void b)
  end


(*===-- Builder -----------------------------------------------------------===*)

let test_builder () =
  let (++) x f = f x; x in

  begin group "parent";
    insist (try
              ignore (insertion_block (builder context));
              false
            with Not_found ->
              true);

    let fty = function_type void_type [| i32_type |] in
    let fn = define_function "BuilderParent" fty m in
    let bb = entry_block fn in
    let b = builder_at_end context bb in
    let p = param fn 0 in
    let sum = build_add p p "sum" b in
    ignore (build_ret_void b);

    insist (fn = block_parent bb);
    insist (fn = param_parent p);
    insist (bb = instr_parent sum);
    insist (bb = insertion_block b)
  end;

  group "ret void";
  begin
    (* CHECK: ret void
     *)
    let fty = function_type void_type [| |] in
    let fn = declare_function "X6" fty m in
    let b = builder_at_end context (append_block context "Bb01" fn) in
    ignore (build_ret_void b)
  end;

  group "ret aggregate";
  begin
      (* CHECK: ret { i8, i64 } { i8 4, i64 5 }
       *)
      let sty = struct_type context [| i8_type; i64_type |] in
      let fty = function_type sty [| |] in
      let fn = declare_function "XA6" fty m in
      let b = builder_at_end context (append_block context "Bb01" fn) in
      let agg = [| const_int i8_type 4; const_int i64_type 5 |] in
      ignore (build_aggregate_ret agg b)
  end;

  (* The rest of the tests will use one big function. *)
  let fty = function_type i32_type [| i32_type; i32_type |] in
  let fn = define_function "X7" fty m in
  let atentry = builder_at_end context (entry_block fn) in
  let p1 = param fn 0 ++ set_value_name "P1" in
  let p2 = param fn 1 ++ set_value_name "P2" in
  let f1 = build_uitofp p1 float_type "F1" atentry in
  let f2 = build_uitofp p2 float_type "F2" atentry in

  let bb00 = append_block context "Bb00" fn in
  ignore (build_unreachable (builder_at_end context bb00));

  group "function attribute";
  begin
    let signext  = create_enum_attr context "signext" 0L in
    let zeroext  = create_enum_attr context "zeroext" 0L in
    let noalias  = create_enum_attr context "noalias" 0L in
    let nounwind = create_enum_attr context "nounwind" 0L in
    let no_sse   = create_string_attr context "no-sse" "" in

    add_function_attr fn signext (AttrIndex.Param 0);
    add_function_attr fn noalias (AttrIndex.Param 1);
    insist ((function_attrs fn (AttrIndex.Param 1)) = [|noalias|]);
    remove_enum_function_attr fn (enum_attr_kind "noalias") (AttrIndex.Param 1);
    add_function_attr fn no_sse (AttrIndex.Param 1);
    insist ((function_attrs fn (AttrIndex.Param 1)) = [|no_sse|]);
    remove_string_function_attr fn "no-sse" (AttrIndex.Param 1);
    insist ((function_attrs fn (AttrIndex.Param 1)) = [||]);
    add_function_attr fn nounwind AttrIndex.Function;
    add_function_attr fn zeroext AttrIndex.Return;

    (* CHECK: define zeroext i32 @X7(i32 signext %P1, i32 %P2)
     *)
  end;

  group "casts"; begin
    let void_ptr = pointer_type i8_type in

    (* CHECK-DAG: %build_trunc = trunc i32 %P1 to i8
     * CHECK-DAG: %build_trunc2 = trunc i32 %P1 to i8
     * CHECK-DAG: %build_trunc3 = trunc i32 %P1 to i8
     * CHECK-DAG: %build_zext = zext i8 %build_trunc to i32
     * CHECK-DAG: %build_zext2 = zext i8 %build_trunc to i32
     * CHECK-DAG: %build_sext = sext i32 %build_zext to i64
     * CHECK-DAG: %build_sext2 = sext i32 %build_zext to i64
     * CHECK-DAG: %build_sext3 = sext i32 %build_zext to i64
     * CHECK-DAG: %build_uitofp = uitofp i64 %build_sext to float
     * CHECK-DAG: %build_sitofp = sitofp i32 %build_zext to double
     * CHECK-DAG: %build_fptoui = fptoui float %build_uitofp to i32
     * CHECK-DAG: %build_fptosi = fptosi double %build_sitofp to i64
     * CHECK-DAG: %build_fptrunc = fptrunc double %build_sitofp to float
     * CHECK-DAG: %build_fptrunc2 = fptrunc double %build_sitofp to float
     * CHECK-DAG: %build_fpext = fpext float %build_fptrunc to double
     * CHECK-DAG: %build_fpext2 = fpext float %build_fptrunc to double
     * CHECK-DAG: %build_inttoptr = inttoptr i32 %P1 to i8*
     * CHECK-DAG: %build_ptrtoint = ptrtoint i8* %build_inttoptr to i64
     * CHECK-DAG: %build_ptrtoint2 = ptrtoint i8* %build_inttoptr to i64
     * CHECK-DAG: %build_bitcast = bitcast i64 %build_ptrtoint to double
     * CHECK-DAG: %build_bitcast2 = bitcast i64 %build_ptrtoint to double
     * CHECK-DAG: %build_bitcast3 = bitcast i64 %build_ptrtoint to double
     * CHECK-DAG: %build_bitcast4 = bitcast i64 %build_ptrtoint to double
     * CHECK-DAG: %build_pointercast = bitcast i8* %build_inttoptr to i16*
     *)
    let inst28 = build_trunc p1 i8_type "build_trunc" atentry in
    let inst29 = build_zext inst28 i32_type "build_zext" atentry in
    let inst30 = build_sext inst29 i64_type "build_sext" atentry in
    let inst31 = build_uitofp inst30 float_type "build_uitofp" atentry in
    let inst32 = build_sitofp inst29 double_type "build_sitofp" atentry in
    ignore(build_fptoui inst31 i32_type "build_fptoui" atentry);
    ignore(build_fptosi inst32 i64_type "build_fptosi" atentry);
    let inst35 = build_fptrunc inst32 float_type "build_fptrunc" atentry in
    ignore(build_fpext inst35 double_type "build_fpext" atentry);
    let inst37 = build_inttoptr p1 void_ptr "build_inttoptr" atentry in
    let inst38 = build_ptrtoint inst37 i64_type "build_ptrtoint" atentry in
    ignore(build_bitcast inst38 double_type "build_bitcast" atentry);
    ignore(build_zext_or_bitcast inst38 double_type "build_bitcast2" atentry);
    ignore(build_sext_or_bitcast inst38 double_type "build_bitcast3" atentry);
    ignore(build_trunc_or_bitcast inst38 double_type "build_bitcast4" atentry);
    ignore(build_pointercast inst37 (pointer_type i16_type) "build_pointercast" atentry);

    ignore(build_zext_or_bitcast inst28 i32_type "build_zext2" atentry);
    ignore(build_sext_or_bitcast inst29 i64_type "build_sext2" atentry);
    ignore(build_trunc_or_bitcast p1 i8_type "build_trunc2" atentry);
    ignore(build_pointercast inst37 i64_type "build_ptrtoint2" atentry);
    ignore(build_intcast inst29 i64_type "build_sext3" atentry);
    ignore(build_intcast p1 i8_type "build_trunc3" atentry);
    ignore(build_fpcast inst35 double_type "build_fpext2" atentry);
    ignore(build_fpcast inst32 float_type "build_fptrunc2" atentry);
  end;

  group "comparisons"; begin
    (* CHECK: %build_icmp_ne = icmp ne i32 %P1, %P2
     * CHECK: %build_icmp_sle = icmp sle i32 %P2, %P1
     * CHECK: %build_fcmp_false = fcmp false float %F1, %F2
     * CHECK: %build_fcmp_true = fcmp true float %F2, %F1
     * CHECK: %build_is_null{{.*}}= icmp eq{{.*}}%X0,{{.*}}null
     * CHECK: %build_is_not_null = icmp ne i8* %X1, null
     * CHECK: %build_ptrdiff
     *)
    let c = build_icmp Icmp.Ne    p1 p2 "build_icmp_ne" atentry in
    insist (Some Icmp.Ne = icmp_predicate c);
    insist (None = fcmp_predicate c);

    let c = build_icmp Icmp.Sle   p2 p1 "build_icmp_sle" atentry in
    insist (Some Icmp.Sle = icmp_predicate c);
    insist (None = fcmp_predicate c);

    let c = build_fcmp Fcmp.False f1 f2 "build_fcmp_false" atentry in
    (* insist (Some Fcmp.False = fcmp_predicate c); *)
    insist (None = icmp_predicate c);

    let c = build_fcmp Fcmp.True  f2 f1 "build_fcmp_true" atentry in
    (* insist (Some Fcmp.True = fcmp_predicate c); *)
    insist (None = icmp_predicate c);

    let g0 = declare_global (pointer_type i8_type) "g0" m in
    let g1 = declare_global (pointer_type i8_type) "g1" m in
    let p0 = build_load g0 "X0" atentry in
    let p1 = build_load g1 "X1" atentry in
    ignore (build_is_null p0 "build_is_null" atentry);
    ignore (build_is_not_null p1 "build_is_not_null" atentry);
    ignore (build_ptrdiff p1 p0 "build_ptrdiff" atentry);
  end;

  group "miscellaneous"; begin
    (* CHECK: %build_call = tail call cc63 zeroext i32 @{{.*}}(i32 signext %P2, i32 %P1)
     * CHECK: %build_select = select i1 %build_icmp, i32 %P1, i32 %P2
     * CHECK: %build_va_arg = va_arg i8** null, i32
     * CHECK: %build_extractelement = extractelement <4 x i32> %Vec1, i32 %P2
     * CHECK: %build_insertelement = insertelement <4 x i32> %Vec1, i32 %P1, i32 %P2
     * CHECK: %build_shufflevector = shufflevector <4 x i32> %Vec1, <4 x i32> %Vec2, <4 x i32> <i32 1, i32 1, i32 0, i32 0>
     * CHECK: %build_insertvalue0 = insertvalue{{.*}}%bl, i32 1, 0
     * CHECK: %build_extractvalue = extractvalue{{.*}}%build_insertvalue1, 1
     *)
    let ci = build_call fn [| p2; p1 |] "build_call" atentry in
    insist (CallConv.c = instruction_call_conv ci);
    set_instruction_call_conv 63 ci;
    insist (63 = instruction_call_conv ci);
    insist (not (is_tail_call ci));
    set_tail_call true ci;
    insist (is_tail_call ci);

    let signext  = create_enum_attr context "signext" 0L in
    let zeroext  = create_enum_attr context "zeroext" 0L in
    let noalias  = create_enum_attr context "noalias" 0L in
    let noreturn = create_enum_attr context "noreturn" 0L in
    let no_sse   = create_string_attr context "no-sse" "" in

    add_call_site_attr ci signext (AttrIndex.Param 0);
    add_call_site_attr ci noalias (AttrIndex.Param 1);
    insist ((call_site_attrs ci (AttrIndex.Param 1)) = [|noalias|]);
    remove_enum_call_site_attr ci (enum_attr_kind "noalias") (AttrIndex.Param 1);
    add_call_site_attr ci no_sse (AttrIndex.Param 1);
    insist ((call_site_attrs ci (AttrIndex.Param 1)) = [|no_sse|]);
    remove_string_call_site_attr ci "no-sse" (AttrIndex.Param 1);
    insist ((call_site_attrs ci (AttrIndex.Param 1)) = [||]);
    add_call_site_attr ci noreturn AttrIndex.Function;
    add_call_site_attr ci zeroext AttrIndex.Return;

    let inst46 = build_icmp Icmp.Eq p1 p2 "build_icmp" atentry in
    ignore (build_select inst46 p1 p2 "build_select" atentry);
    ignore (build_va_arg
      (const_null (pointer_type (pointer_type i8_type)))
      i32_type "build_va_arg" atentry);

    (* Set up some vector vregs. *)
    let one  = const_int i32_type 1 in
    let zero = const_int i32_type 0 in
    let t1 = const_vector [| one; zero; one; zero |] in
    let t2 = const_vector [| zero; one; zero; one |] in
    let t3 = const_vector [| one; one; zero; zero |] in
    let vec1 = build_insertelement t1 p1 p2 "Vec1" atentry in
    let vec2 = build_insertelement t2 p1 p2 "Vec2" atentry in
    let sty = struct_type context [| i32_type; i8_type |] in

    ignore (build_extractelement vec1 p2 "build_extractelement" atentry);
    ignore (build_insertelement vec1 p1 p2 "build_insertelement" atentry);
    ignore (build_shufflevector vec1 vec2 t3 "build_shufflevector" atentry);

    let p = build_alloca sty "ba" atentry in
    let agg = build_load p "bl" atentry in
    let agg0 = build_insertvalue agg (const_int i32_type 1) 0
                 "build_insertvalue0" atentry in
    let agg1 = build_insertvalue agg0 (const_int i8_type 2) 1
                 "build_insertvalue1" atentry in
    ignore (build_extractvalue agg1 1 "build_extractvalue" atentry)
  end;

  group "metadata"; begin
    (* CHECK: %metadata = add i32 %P1, %P2, !test !1
     * !1 is metadata emitted at EOF.
     *)
    let i = build_add p1 p2 "metadata" atentry in
    insist ((has_metadata i) = false);

    let m1 = const_int i32_type 1 in
    let m2 = mdstring context "metadata test" in
    let md = mdnode context [| m1; m2 |] in

    let kind = mdkind_id context "test" in
    set_metadata i kind md;

    insist ((has_metadata i) = true);
    insist ((metadata i kind) = Some md);
    insist ((get_mdnode_operands md) = [| m1; m2 |]);

    clear_metadata i kind;

    insist ((has_metadata i) = false);
    insist ((metadata i kind) = None);

    set_metadata i kind md
  end;

  group "named metadata"; begin
    (* !llvm.module.flags is emitted at EOF. *)
    let n1 = const_int i32_type 1 in
    let n2 = mdstring context "Debug Info Version" in
    let n3 = const_int i32_type 3 in
    let md = mdnode context [| n1; n2; n3 |] in
    add_named_metadata_operand m "llvm.module.flags" md;

    insist ((get_named_metadata m "llvm.module.flags") = [| md |])
  end;

  group "ret"; begin
    (* CHECK: ret{{.*}}P1
     *)
    let ret = build_ret p1 atentry in
    position_before ret atentry
  end;

  (* see test/Feature/exception.ll *)
  let bblpad = append_block context "Bblpad" fn in
  let rt = struct_type context [| pointer_type i8_type; i32_type |] in
  let ft = var_arg_function_type i32_type  [||] in
  let personality = declare_function "__gxx_personality_v0" ft m in
  let ztic = declare_global (pointer_type i8_type) "_ZTIc" m in
  let ztid = declare_global (pointer_type i8_type) "_ZTId" m in
  let ztipkc = declare_global (pointer_type i8_type) "_ZTIPKc" m in
  begin
      set_global_constant true ztic;
      set_global_constant true ztid;
      set_global_constant true ztipkc;
      let lp = build_landingpad rt personality 0 "lpad"
       (builder_at_end context bblpad) in begin
           set_cleanup lp true;
           add_clause lp ztic;
           insist((pointer_type (pointer_type i8_type)) = type_of ztid);
           let ety = pointer_type (pointer_type i8_type) in
           add_clause lp (const_array ety [| ztipkc; ztid |]);
           ignore (build_resume lp (builder_at_end context bblpad));
      end;
      (* CHECK: landingpad
       * CHECK: cleanup
       * CHECK: catch{{.*}}i8**{{.*}}@_ZTIc
       * CHECK: filter{{.*}}@_ZTIPKc{{.*}}@_ZTId
       * CHECK: resume
       * *)
  end;

  group "br"; begin
    (* CHECK: br{{.*}}Bb02
     *)
    let bb02 = append_block context "Bb02" fn in
    let b = builder_at_end context bb02 in
    let br = build_br bb02 b in
    insist (successors br = [| bb02 |]) ;
    insist (is_conditional br = false) ;
    insist (get_branch br = Some (`Unconditional bb02)) ;
  end;

  group "cond_br"; begin
    (* CHECK: br{{.*}}build_br{{.*}}Bb03{{.*}}Bb00
     *)
    let bb03 = append_block context "Bb03" fn in
    let b = builder_at_end context bb03 in
    let cond = build_trunc p1 i1_type "build_br" b in
    let br = build_cond_br cond bb03 bb00 b in
    insist (num_successors br = 2) ;
    insist (successor br 0 = bb03) ;
    insist (successor br 1 = bb00) ;
    insist (is_conditional br = true) ;
    insist (get_branch br = Some (`Conditional (cond, bb03, bb00))) ;
  end;

  group "switch"; begin
    (* CHECK: switch{{.*}}P1{{.*}}SwiBlock3
     * CHECK: 2,{{.*}}SwiBlock2
     *)
    let bb1 = append_block context "SwiBlock1" fn in
    let bb2 = append_block context "SwiBlock2" fn in
    ignore (build_unreachable (builder_at_end context bb2));
    let bb3 = append_block context "SwiBlock3" fn in
    ignore (build_unreachable (builder_at_end context bb3));
    let si = build_switch p1 bb3 1 (builder_at_end context bb1) in begin
        ignore (add_case si (const_int i32_type 2) bb2);
        insist (switch_default_dest si = bb3);
    end;
    insist (num_successors si = 2) ;
    insist (get_branch si = None) ;
  end;

  group "malloc/free"; begin
      (* CHECK: call{{.*}}@malloc(i32 ptrtoint
       * CHECK: call{{.*}}@free(i8*
       * CHECK: call{{.*}}@malloc(i32 %
       *)
      let bb1 = append_block context "MallocBlock1" fn in
      let m1 = (build_malloc (pointer_type i32_type) "m1"
      (builder_at_end context bb1)) in
      ignore (build_free m1 (builder_at_end context bb1));
      ignore (build_array_malloc i32_type p1 "m2" (builder_at_end context bb1));
      ignore (build_unreachable (builder_at_end context bb1));
  end;

  group "indirectbr"; begin
    (* CHECK: indirectbr i8* blockaddress(@X7, %IBRBlock2), [label %IBRBlock2, label %IBRBlock3]
     *)
    let bb1 = append_block context "IBRBlock1" fn in

    let bb2 = append_block context "IBRBlock2" fn in
    ignore (build_unreachable (builder_at_end context bb2));

    let bb3 = append_block context "IBRBlock3" fn in
    ignore (build_unreachable (builder_at_end context bb3));

    let addr = block_address fn bb2 in
    let ibr = build_indirect_br addr 2 (builder_at_end context bb1) in
    ignore (add_destination ibr bb2);
    ignore (add_destination ibr bb3)
  end;

  group "invoke"; begin
    (* CHECK: build_invoke{{.*}}invoke{{.*}}P1{{.*}}P2
     * CHECK: to{{.*}}Bb04{{.*}}unwind{{.*}}Bblpad
     *)
    let bb04 = append_block context "Bb04" fn in
    let b = builder_at_end context bb04 in
    ignore (build_invoke fn [| p1; p2 |] bb04 bblpad "build_invoke" b)
  end;

  group "unreachable"; begin
    (* CHECK: unreachable
     *)
    let bb06 = append_block context "Bb06" fn in
    let b = builder_at_end context bb06 in
    ignore (build_unreachable b)
  end;

  group "arithmetic"; begin
    let bb07 = append_block context "Bb07" fn in
    let b = builder_at_end context bb07 in

    (* CHECK: %build_add = add i32 %P1, %P2
     * CHECK: %build_nsw_add = add nsw i32 %P1, %P2
     * CHECK: %build_nuw_add = add nuw i32 %P1, %P2
     * CHECK: %build_fadd = fadd float %F1, %F2
     * CHECK: %build_sub = sub i32 %P1, %P2
     * CHECK: %build_nsw_sub = sub nsw i32 %P1, %P2
     * CHECK: %build_nuw_sub = sub nuw i32 %P1, %P2
     * CHECK: %build_fsub = fsub float %F1, %F2
     * CHECK: %build_mul = mul i32 %P1, %P2
     * CHECK: %build_nsw_mul = mul nsw i32 %P1, %P2
     * CHECK: %build_nuw_mul = mul nuw i32 %P1, %P2
     * CHECK: %build_fmul = fmul float %F1, %F2
     * CHECK: %build_udiv = udiv i32 %P1, %P2
     * CHECK: %build_sdiv = sdiv i32 %P1, %P2
     * CHECK: %build_exact_sdiv = sdiv exact i32 %P1, %P2
     * CHECK: %build_fdiv = fdiv float %F1, %F2
     * CHECK: %build_urem = urem i32 %P1, %P2
     * CHECK: %build_srem = srem i32 %P1, %P2
     * CHECK: %build_frem = frem float %F1, %F2
     * CHECK: %build_shl = shl i32 %P1, %P2
     * CHECK: %build_lshl = lshr i32 %P1, %P2
     * CHECK: %build_ashl = ashr i32 %P1, %P2
     * CHECK: %build_and = and i32 %P1, %P2
     * CHECK: %build_or = or i32 %P1, %P2
     * CHECK: %build_xor = xor i32 %P1, %P2
     * CHECK: %build_neg = sub i32 0, %P1
     * CHECK: %build_nsw_neg = sub nsw i32 0, %P1
     * CHECK: %build_nuw_neg = sub nuw i32 0, %P1
     * CHECK: %build_fneg = fneg float %F1
     * CHECK: %build_not = xor i32 %P1, -1
     * CHECK: %build_freeze = freeze i32 %P1
     *)
    ignore (build_add p1 p2 "build_add" b);
    ignore (build_nsw_add p1 p2 "build_nsw_add" b);
    ignore (build_nuw_add p1 p2 "build_nuw_add" b);
    ignore (build_fadd f1 f2 "build_fadd" b);
    ignore (build_sub p1 p2 "build_sub" b);
    ignore (build_nsw_sub p1 p2 "build_nsw_sub" b);
    ignore (build_nuw_sub p1 p2 "build_nuw_sub" b);
    ignore (build_fsub f1 f2 "build_fsub" b);
    ignore (build_mul p1 p2 "build_mul" b);
    ignore (build_nsw_mul p1 p2 "build_nsw_mul" b);
    ignore (build_nuw_mul p1 p2 "build_nuw_mul" b);
    ignore (build_fmul f1 f2 "build_fmul" b);
    ignore (build_udiv p1 p2 "build_udiv" b);
    ignore (build_sdiv p1 p2 "build_sdiv" b);
    ignore (build_exact_sdiv p1 p2 "build_exact_sdiv" b);
    ignore (build_fdiv f1 f2 "build_fdiv" b);
    ignore (build_urem p1 p2 "build_urem" b);
    ignore (build_srem p1 p2 "build_srem" b);
    ignore (build_frem f1 f2 "build_frem" b);
    ignore (build_shl p1 p2 "build_shl" b);
    ignore (build_lshr p1 p2 "build_lshl" b);
    ignore (build_ashr p1 p2 "build_ashl" b);
    ignore (build_and p1 p2 "build_and" b);
    ignore (build_or p1 p2 "build_or" b);
    ignore (build_xor p1 p2 "build_xor" b);
    ignore (build_neg p1 "build_neg" b);
    ignore (build_nsw_neg p1 "build_nsw_neg" b);
    ignore (build_nuw_neg p1 "build_nuw_neg" b);
    ignore (build_fneg f1 "build_fneg" b);
    ignore (build_not p1 "build_not" b);
    ignore (build_freeze p1 "build_freeze" b);
    ignore (build_unreachable b)
  end;

  group "memory"; begin
    let bb08 = append_block context "Bb08" fn in
    let b = builder_at_end context bb08 in

    (* CHECK: %build_alloca = alloca i32
     * CHECK: %build_array_alloca = alloca i32, i32 %P2
     * CHECK: %build_load = load volatile i32, i32* %build_array_alloca, align 4
     * CHECK: store volatile i32 %P2, i32* %build_alloca, align 4
     * CHECK: %build_gep = getelementptr i32, i32* %build_array_alloca, i32 %P2
     * CHECK: %build_in_bounds_gep = getelementptr inbounds i32, i32* %build_array_alloca, i32 %P2
     * CHECK: %build_struct_gep = getelementptr inbounds{{.*}}%build_alloca2, i32 0, i32 1
     * CHECK: %build_atomicrmw = atomicrmw xchg i8* %p, i8 42 seq_cst
     *)
    let alloca = build_alloca i32_type "build_alloca" b in
    let array_alloca = build_array_alloca i32_type p2 "build_array_alloca" b in

    let load = build_load array_alloca "build_load" b in
    ignore(set_alignment 4 load);
    ignore(set_volatile true load);
    insist(true = is_volatile load);
    insist(4 = alignment load);

    let store = build_store p2 alloca b in
    ignore(set_volatile true store);
    ignore(set_alignment 4 store);
    insist(true = is_volatile store);
    insist(4 = alignment store);
    ignore(build_gep array_alloca [| p2 |] "build_gep" b);
    ignore(build_in_bounds_gep array_alloca [| p2 |] "build_in_bounds_gep" b);

    let sty = struct_type context [| i32_type; i8_type |] in
    let alloca2 = build_alloca sty "build_alloca2" b in
    ignore(build_struct_gep alloca2 1 "build_struct_gep" b);

    let p = build_alloca i8_type "p" b in
    ignore(build_atomicrmw AtomicRMWBinOp.Xchg p (const_int i8_type 42)
              AtomicOrdering.SequentiallyConsistent false "build_atomicrmw"
              b);

    ignore(build_unreachable b)
  end;

  group "string"; begin
    let bb09 = append_block context "Bb09" fn in
    let b = builder_at_end context bb09 in
    let p = build_alloca (pointer_type i8_type) "p" b in
    (* build_global_string is emitted above.
     * CHECK: store{{.*}}build_global_string1{{.*}}p
     * *)
    ignore (build_global_string "stringval" "build_global_string" b);
    let g = build_global_stringptr "stringval" "build_global_string1" b in
    ignore (build_store g p b);
    ignore(build_unreachable b);
  end;

  group "phi"; begin
    (* CHECK: PhiNode{{.*}}P1{{.*}}PhiBlock1{{.*}}P2{{.*}}PhiBlock2
     *)
    let b1 = append_block context "PhiBlock1" fn in
    let b2 = append_block context "PhiBlock2" fn in

    let jb = append_block context "PhiJoinBlock" fn in
    ignore (build_br jb (builder_at_end context b1));
    ignore (build_br jb (builder_at_end context b2));
    let at_jb = builder_at_end context jb in

    let phi = build_phi [(p1, b1)] "PhiNode" at_jb in
    insist ([(p1, b1)] = incoming phi);

    add_incoming (p2, b2) phi;
    insist ([(p1, b1); (p2, b2)] = incoming phi);

    (* CHECK: %PhiEmptyNode = phi i8
     *)
    let phi_empty = build_empty_phi i8_type "PhiEmptyNode" at_jb in
    insist ([] = incoming phi_empty);

    (* can't emit an empty phi to bitcode *)
    add_incoming (const_int i8_type 1, b1) phi_empty;
    add_incoming (const_int i8_type 2, b2) phi_empty;

    ignore (build_unreachable at_jb);
  end

(* End-of-file checks for things like metdata and attributes.
 * CHECK: !llvm.module.flags = !{!0}
 * CHECK: !0 = !{i32 1, !"Debug Info Version", i32 3}
 * CHECK: !1 = !{i32 1, !"metadata test"}
 *)

(*===-- Pass Managers -----------------------------------------------------===*)

let test_pass_manager () =
  let (++) x f = ignore (f x); x in

  begin group "module pass manager";
    ignore (PassManager.create ()
             ++ PassManager.run_module m
             ++ PassManager.dispose)
  end;

  begin group "function pass manager";
    let fty = function_type void_type [| |] in
    let fn = define_function "FunctionPassManager" fty m in
    ignore (build_ret_void (builder_at_end context (entry_block fn)));

    ignore (PassManager.create_function m
             ++ PassManager.initialize
             ++ PassManager.run_function fn
             ++ PassManager.finalize
             ++ PassManager.dispose)
  end


(*===-- Memory Buffer -----------------------------------------------------===*)

let test_memory_buffer () =
  group "memory buffer";
  let buf = MemoryBuffer.of_string "foobar" in
  insist ((MemoryBuffer.as_string buf) = "foobar")


(*===-- Writer ------------------------------------------------------------===*)

let test_writer () =
  group "valid";
  insist (match Llvm_analysis.verify_module m with
          | None -> true
          | Some msg -> prerr_string msg; false);

  group "writer";
  insist (write_bitcode_file m filename);

  dispose_module m


(*===-- Driver ------------------------------------------------------------===*)

let _ =
  suite "contained types"  test_contained_types;
  suite "conversion"       test_conversion;
  suite "target"           test_target;
  suite "constants"        test_constants;
  suite "attributes"       test_attributes;
  suite "global values"    test_global_values;
  suite "global variables" test_global_variables;
  suite "uses"             test_uses;
  suite "users"            test_users;
  suite "aliases"          test_aliases;
  suite "functions"        test_functions;
  suite "params"           test_params;
  suite "basic blocks"     test_basic_blocks;
  suite "instructions"     test_instructions;
  suite "builder"          test_builder;
  suite "pass manager"     test_pass_manager;
  suite "memory buffer"    test_memory_buffer;
  suite "writer"           test_writer; (* Keep this last; it disposes m. *)
  exit !exit_status
