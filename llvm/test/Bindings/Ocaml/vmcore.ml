(* RUN: %ocamlopt -warn-error A llvm.cmxa llvm_analysis.cmxa llvm_bitwriter.cmxa %s -o %t
 * RUN: %t %t.bc
 * RUN: llvm-dis < %t.bc > %t.ll
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


(*===-- Target ------------------------------------------------------------===*)

let test_target () =
  begin group "triple";
    (* RUN: grep "i686-apple-darwin8" < %t.ll
     *)
    let trip = "i686-apple-darwin8" in
    set_target_triple trip m;
    insist (trip = target_triple m)
  end;
  
  begin group "layout";
    (* RUN: grep "bogus" < %t.ll
     *)
    let layout = "bogus" in
    set_data_layout layout m;
    insist (layout = data_layout m)
  end


(*===-- Constants ---------------------------------------------------------===*)

let test_constants () =
  (* RUN: grep "const_int.*i32.*-1" < %t.ll
   *)
  group "int";
  let c = const_int i32_type (-1) in
  ignore (define_global "const_int" c m);
  insist (i32_type = type_of c);
  insist (is_constant c);

  (* RUN: grep "const_sext_int.*i64.*-1" < %t.ll
   *)
  group "sext int";
  let c = const_int i64_type (-1) in
  ignore (define_global "const_sext_int" c m);
  insist (i64_type = type_of c);

  (* RUN: grep "const_zext_int64.*i64.*4294967295" < %t.ll
   *)
  group "zext int64";
  let c = const_of_int64 i64_type (Int64.of_string "4294967295") false in
  ignore (define_global "const_zext_int64" c m);
  insist (i64_type = type_of c);

  (* RUN: grep "const_int_string.*i32.*-1" < %t.ll
   *)
  group "int string";
  let c = const_int_of_string i32_type "-1" 10 in
  ignore (define_global "const_int_string" c m);
  insist (i32_type = type_of c);

  (* RUN: grep 'const_string.*"cruel\\00world"' < %t.ll
   *)
  group "string";
  let c = const_string context "cruel\000world" in
  ignore (define_global "const_string" c m);
  insist ((array_type i8_type 11) = type_of c);

  (* RUN: grep 'const_stringz.*"hi\\00again\\00"' < %t.ll
   *)
  group "stringz";
  let c = const_stringz context "hi\000again" in
  ignore (define_global "const_stringz" c m);
  insist ((array_type i8_type 9) = type_of c);

  (* RUN: grep "const_single.*2.75" < %t.ll
   * RUN: grep "const_double.*3.1459" < %t.ll
   * RUN: grep "const_double_string.*1.25" < %t.ll
   *)
  begin group "real";
    let cs = const_float float_type 2.75 in
    ignore (define_global "const_single" cs m);
    insist (float_type = type_of cs);
    
    let cd = const_float double_type 3.1459 in
    ignore (define_global "const_double" cd m);
    insist (double_type = type_of cd);

    let cd = const_float_of_string double_type "1.25" in
    ignore (define_global "const_double_string" cd m);
    insist (double_type = type_of cd)
  end;
  
  let one = const_int i16_type 1 in
  let two = const_int i16_type 2 in
  let three = const_int i32_type 3 in
  let four = const_int i32_type 4 in
  
  (* RUN: grep "const_array.*[i32 3, i32 4]" < %t.ll
   *)
  group "array";
  let c = const_array i32_type [| three; four |] in
  ignore (define_global "const_array" c m);
  insist ((array_type i32_type 2) = (type_of c));
  
  (* RUN: grep "const_vector.*<i16 1, i16 2.*>" < %t.ll
   *)
  group "vector";
  let c = const_vector [| one; two; one; two;
                          one; two; one; two |] in
  ignore (define_global "const_vector" c m);
  insist ((vector_type i16_type 8) = (type_of c));

  (* RUN: grep "const_structure.*.i16 1, i16 2, i32 3, i32 4" < %t.ll
   *)
  group "structure";
  let c = const_struct context [| one; two; three; four |] in
  ignore (define_global "const_structure" c m);
  insist ((struct_type context [| i16_type; i16_type; i32_type; i32_type |])
        = (type_of c));

  (* RUN: grep "const_null.*zeroinit" < %t.ll
   *)
  group "null";
  let c = const_null (packed_struct_type context [| i1_type; i8_type; i64_type;
                                                    double_type |]) in
  ignore (define_global "const_null" c m);
  
  (* RUN: grep "const_all_ones.*-1" < %t.ll
   *)
  group "all ones";
  let c = const_all_ones i64_type in
  ignore (define_global "const_all_ones" c m);

  group "pointer null"; begin
    (* RUN: grep "const_pointer_null = global i64\* null" < %t.ll
     *)
    let c = const_pointer_null (pointer_type i64_type) in
    ignore (define_global "const_pointer_null" c m);
  end;
  
  (* RUN: grep "const_undef.*undef" < %t.ll
   *)
  group "undef";
  let c = undef i1_type in
  ignore (define_global "const_undef" c m);
  insist (i1_type = type_of c);
  insist (is_undef c);
  
  group "constant arithmetic";
  (* RUN: grep "@const_neg = global i64 sub" < %t.ll
   * RUN: grep "@const_nsw_neg = global i64 sub nsw " < %t.ll
   * RUN: grep "@const_nuw_neg = global i64 sub nuw " < %t.ll
   * RUN: grep "@const_fneg = global double fsub " < %t.ll
   * RUN: grep "@const_not = global i64 xor " < %t.ll
   * RUN: grep "@const_add = global i64 add " < %t.ll
   * RUN: grep "@const_nsw_add = global i64 add nsw " < %t.ll
   * RUN: grep "@const_nuw_add = global i64 add nuw " < %t.ll
   * RUN: grep "@const_fadd = global double fadd " < %t.ll
   * RUN: grep "@const_sub = global i64 sub " < %t.ll
   * RUN: grep "@const_nsw_sub = global i64 sub nsw " < %t.ll
   * RUN: grep "@const_nuw_sub = global i64 sub nuw " < %t.ll
   * RUN: grep "@const_fsub = global double fsub " < %t.ll
   * RUN: grep "@const_mul = global i64 mul " < %t.ll
   * RUN: grep "@const_nsw_mul = global i64 mul nsw " < %t.ll
   * RUN: grep "@const_nuw_mul = global i64 mul nuw " < %t.ll
   * RUN: grep "@const_fmul = global double fmul " < %t.ll
   * RUN: grep "@const_udiv = global i64 udiv " < %t.ll
   * RUN: grep "@const_sdiv = global i64 sdiv " < %t.ll
   * RUN: grep "@const_exact_sdiv = global i64 sdiv exact " < %t.ll
   * RUN: grep "@const_fdiv = global double fdiv " < %t.ll
   * RUN: grep "@const_urem = global i64 urem " < %t.ll
   * RUN: grep "@const_srem = global i64 srem " < %t.ll
   * RUN: grep "@const_frem = global double frem " < %t.ll
   * RUN: grep "@const_and = global i64 and " < %t.ll
   * RUN: grep "@const_or = global i64 or " < %t.ll
   * RUN: grep "@const_xor = global i64 xor " < %t.ll
   * RUN: grep "@const_icmp = global i1 icmp sle " < %t.ll
   * RUN: grep "@const_fcmp = global i1 fcmp ole " < %t.ll
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
  (* RUN: grep "const_trunc.*trunc" < %t.ll
   * RUN: grep "const_sext.*sext" < %t.ll
   * RUN: grep "const_zext.*zext" < %t.ll
   * RUN: grep "const_fptrunc.*fptrunc" < %t.ll
   * RUN: grep "const_fpext.*fpext" < %t.ll
   * RUN: grep "const_uitofp.*uitofp" < %t.ll
   * RUN: grep "const_sitofp.*sitofp" < %t.ll
   * RUN: grep "const_fptoui.*fptoui" < %t.ll
   * RUN: grep "const_fptosi.*fptosi" < %t.ll
   * RUN: grep "const_ptrtoint.*ptrtoint" < %t.ll
   * RUN: grep "const_inttoptr.*inttoptr" < %t.ll
   * RUN: grep "const_bitcast.*bitcast" < %t.ll
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
  
  group "misc constants";
  (* RUN: grep "const_size_of.*getelementptr.*null" < %t.ll
   * RUN: grep "const_gep.*getelementptr" < %t.ll
   * RUN: grep "const_select.*select" < %t.ll
   * RUN: grep "const_extractelement.*extractelement" < %t.ll
   * RUN: grep "const_insertelement.*insertelement" < %t.ll
   * RUN: grep "const_shufflevector = global <4 x i32> <i32 0, i32 1, i32 1, i32 0>" < %t.ll
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


(*===-- Global Values -----------------------------------------------------===*)

let test_global_values () =
  let (++) x f = f x; x in
  let zero32 = const_null i32_type in

  (* RUN: grep "GVal01" < %t.ll
   *)
  group "naming";
  let g = define_global "TEMPORARY" zero32 m in
  insist ("TEMPORARY" = value_name g);
  set_value_name "GVal01" g;
  insist ("GVal01" = value_name g);

  (* RUN: grep "GVal02.*linkonce" < %t.ll
   *)
  group "linkage";
  let g = define_global "GVal02" zero32 m ++
          set_linkage Linkage.Link_once in
  insist (Linkage.Link_once = linkage g);

  (* RUN: grep "GVal03.*Hanalei" < %t.ll
   *)
  group "section";
  let g = define_global "GVal03" zero32 m ++
          set_section "Hanalei" in
  insist ("Hanalei" = section g);
  
  (* RUN: grep "GVal04.*hidden" < %t.ll
   *)
  group "visibility";
  let g = define_global "GVal04" zero32 m ++
          set_visibility Visibility.Hidden in
  insist (Visibility.Hidden = visibility g);
  
  (* RUN: grep "GVal05.*align 128" < %t.ll
   *)
  group "alignment";
  let g = define_global "GVal05" zero32 m ++
          set_alignment 128 in
  insist (128 = alignment g)


(*===-- Global Variables --------------------------------------------------===*)

let test_global_variables () =
  let (++) x f = f x; x in
  let fourty_two32 = const_int i32_type 42 in

  group "declarations"; begin
    (* RUN: grep "GVar01.*external" < %t.ll
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
    (* RUN: grep "GVar02.*42" < %t.ll
     * RUN: grep "GVar03.*42" < %t.ll
     *)
    let g = define_global "GVar02" fourty_two32 m in
    let g2 = declare_global i32_type "GVar03" m ++
           set_initializer fourty_two32 in
    insist (not (is_declaration g));
    insist (not (is_declaration g2));
    insist ((global_initializer g) == (global_initializer g2));

    let g = define_qualified_global "QGVar02" fourty_two32 3 m in
    let g2 = declare_qualified_global i32_type "QGVar03" 3 m ++
           set_initializer fourty_two32 in
    insist (not (is_declaration g));
    insist (not (is_declaration g2));
    insist ((global_initializer g) == (global_initializer g2));
  end;

  (* RUN: grep "GVar04.*thread_local" < %t.ll
   *)
  group "threadlocal";
  let g = define_global "GVar04" fourty_two32 m ++
          set_thread_local true in
  insist (is_thread_local g);

  (* RUN: grep -v "GVar05" < %t.ll
   *)
  group "delete";
  let g = define_global "GVar05" fourty_two32 m in
  delete_global g;

  (* RUN: grep -v "ConstGlobalVar.*constant" < %t.ll
   *)
  group "constant";
  let g = define_global "ConstGlobalVar" fourty_two32 m in
  insist (not (is_global_constant g));
  set_global_constant true g;
  insist (is_global_constant g);
  
  begin group "iteration";
    let m = create_module context "temp" in
    
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
  (* RUN: grep "@alias = alias i32\* @aliasee" < %t.ll
   *)
  let v = declare_global i32_type "aliasee" m in
  ignore (add_alias m (pointer_type i32_type) v "alias")


(*===-- Functions ---------------------------------------------------------===*)

let test_functions () =
  let ty = function_type i32_type [| i32_type; i64_type |] in
  let ty2 = function_type i8_type [| i8_type; i64_type |] in
  
  (* RUN: grep 'declare i32 @Fn1(i32, i64)' < %t.ll
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
  
  (* RUN: grep -v "Fn2" < %t.ll
   *)
  group "delete";
  let fn = declare_function "Fn2" ty m in
  delete_function fn;
  
  (* RUN: grep "define.*Fn3" < %t.ll
   *)
  group "define";
  let fn = define_function "Fn3" ty m in
  insist (not (is_declaration fn));
  insist (1 = Array.length (basic_blocks fn));
  ignore (build_unreachable (builder_at_end context (entry_block fn)));
  
  (* RUN: grep "define.*Fn4.*Param1.*Param2" < %t.ll
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
  
  (* RUN: grep "fastcc.*Fn5" < %t.ll
   *)
  group "callconv";
  let fn = define_function "Fn5" ty m in
  insist (CallConv.c = function_call_conv fn);
  set_function_call_conv CallConv.fast fn;
  insist (CallConv.fast = function_call_conv fn);
  ignore (build_unreachable (builder_at_end context (entry_block fn)));
  
  begin group "gc";
    (* RUN: grep "Fn6.*gc.*shadowstack" < %t.ll
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
    add_param_attr p1 Attribute.Sext;
    add_param_attr p2 Attribute.Noalias;
    remove_param_attr p2 Attribute.Noalias;
    add_function_attr f Attribute.Nounwind;
    add_function_attr f Attribute.Noreturn;
    remove_function_attr f Attribute.Noreturn;

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
  
  (* RUN: grep "Bb1" < %t.ll
   *)
  group "entry";
  let fn = declare_function "X" ty m in
  let bb = append_block context "Bb1" fn in
  insist (bb = entry_block fn);
  ignore (build_unreachable (builder_at_end context bb));
  
  (* RUN: grep -v Bb2 < %t.ll
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
  
  (* RUN: grep Bb3 < %t.ll
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
    (* RUN: grep "ret void" < %t.ll
     *)
    let fty = function_type void_type [| |] in
    let fn = declare_function "X6" fty m in
    let b = builder_at_end context (append_block context "Bb01" fn) in
    ignore (build_ret_void b)
  end;

  group "ret aggregate";
  begin
      (* RUN: grep "ret { i8, i64 } { i8 4, i64 5 }" < %t.ll
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
      ignore (add_function_attr fn Attribute.UWTable);
      (* RUN: grep "X7.*uwtable" < %t.ll
       *)
      insist ([Attribute.UWTable] = function_attr fn);
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
      (* RUN: grep "landingpad.*personality.*__gxx_personality_v0" < %t.ll
       * RUN: grep "cleanup" < %t.ll
       * RUN: grep "catch.*i8\*\*.*@_ZTIc" < %t.ll
       * RUN: grep "filter.*@_ZTIPKc.*@_ZTId" < %t.ll
       * RUN: grep "resume " < %t.ll
       * *)
  end;

  group "ret"; begin
    (* RUN: grep "ret.*P1" < %t.ll
     *)
    let ret = build_ret p1 atentry in
    position_before ret atentry
  end;
  
  group "br"; begin
    (* RUN: grep "br.*Bb02" < %t.ll
     *)
    let bb02 = append_block context "Bb02" fn in
    let b = builder_at_end context bb02 in
    ignore (build_br bb02 b)
  end;
  
  group "cond_br"; begin
    (* RUN: grep "br.*build_br.*Bb03.*Bb00" < %t.ll
     *)
    let bb03 = append_block context "Bb03" fn in
    let b = builder_at_end context bb03 in
    let cond = build_trunc p1 i1_type "build_br" b in
    ignore (build_cond_br cond bb03 bb00 b)
  end;
  
  group "switch"; begin
    (* RUN: grep "switch.*P1.*SwiBlock3" < %t.ll
     * RUN: grep "2,.*SwiBlock2" < %t.ll
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
  end;

  group "malloc/free"; begin
      (* RUN: grep "call.*@malloc(i32 ptrtoint" < %t.ll
       * RUN: grep "call.*@free(i8\*" < %t.ll
       * RUN: grep "call.*@malloc(i32 %" < %t.ll
       *)
      let bb1 = append_block context "MallocBlock1" fn in
      let m1 = (build_malloc (pointer_type i32_type) "m1"
      (builder_at_end context bb1)) in
      ignore (build_free m1 (builder_at_end context bb1));
      ignore (build_array_malloc i32_type p1 "m2" (builder_at_end context bb1));
      ignore (build_unreachable (builder_at_end context bb1));
  end;

  group "indirectbr"; begin
    (* RUN: grep "indirectbr i8\* blockaddress(@X7, %IBRBlock2), \[label %IBRBlock2, label %IBRBlock3\]" < %t.ll
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
    (* RUN: grep "build_invoke.*invoke.*P1.*P2" < %t.ll
     * RUN: grep "to.*Bb04.*unwind.*Bblpad" < %t.ll
     *)
    let bb04 = append_block context "Bb04" fn in
    let b = builder_at_end context bb04 in
    ignore (build_invoke fn [| p1; p2 |] bb04 bblpad "build_invoke" b)
  end;
  
  group "unreachable"; begin
    (* RUN: grep "unreachable" < %t.ll
     *)
    let bb06 = append_block context "Bb06" fn in
    let b = builder_at_end context bb06 in
    ignore (build_unreachable b)
  end;
  
  group "arithmetic"; begin
    let bb07 = append_block context "Bb07" fn in
    let b = builder_at_end context bb07 in
    
    (* RUN: grep "%build_add = add i32 %P1, %P2" < %t.ll
     * RUN: grep "%build_nsw_add = add nsw i32 %P1, %P2" < %t.ll
     * RUN: grep "%build_nuw_add = add nuw i32 %P1, %P2" < %t.ll
     * RUN: grep "%build_fadd = fadd float %F1, %F2" < %t.ll
     * RUN: grep "%build_sub = sub i32 %P1, %P2" < %t.ll
     * RUN: grep "%build_nsw_sub = sub nsw i32 %P1, %P2" < %t.ll
     * RUN: grep "%build_nuw_sub = sub nuw i32 %P1, %P2" < %t.ll
     * RUN: grep "%build_fsub = fsub float %F1, %F2" < %t.ll
     * RUN: grep "%build_mul = mul i32 %P1, %P2" < %t.ll
     * RUN: grep "%build_nsw_mul = mul nsw i32 %P1, %P2" < %t.ll
     * RUN: grep "%build_nuw_mul = mul nuw i32 %P1, %P2" < %t.ll
     * RUN: grep "%build_fmul = fmul float %F1, %F2" < %t.ll
     * RUN: grep "%build_udiv = udiv i32 %P1, %P2" < %t.ll
     * RUN: grep "%build_sdiv = sdiv i32 %P1, %P2" < %t.ll
     * RUN: grep "%build_exact_sdiv = sdiv exact i32 %P1, %P2" < %t.ll
     * RUN: grep "%build_fdiv = fdiv float %F1, %F2" < %t.ll
     * RUN: grep "%build_urem = urem i32 %P1, %P2" < %t.ll
     * RUN: grep "%build_srem = srem i32 %P1, %P2" < %t.ll
     * RUN: grep "%build_frem = frem float %F1, %F2" < %t.ll
     * RUN: grep "%build_shl = shl i32 %P1, %P2" < %t.ll
     * RUN: grep "%build_lshl = lshr i32 %P1, %P2" < %t.ll
     * RUN: grep "%build_ashl = ashr i32 %P1, %P2" < %t.ll
     * RUN: grep "%build_and = and i32 %P1, %P2" < %t.ll
     * RUN: grep "%build_or = or i32 %P1, %P2" < %t.ll
     * RUN: grep "%build_xor = xor i32 %P1, %P2" < %t.ll
     * RUN: grep "%build_neg = sub i32 0, %P1" < %t.ll
     * RUN: grep "%build_nsw_neg = sub nsw i32 0, %P1" < %t.ll
     * RUN: grep "%build_nuw_neg = sub nuw i32 0, %P1" < %t.ll
     * RUN: grep "%build_fneg = fsub float .*0.*, %F1" < %t.ll
     * RUN: grep "%build_not = xor i32 %P1, -1" < %t.ll
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
    ignore (build_unreachable b)
  end;
  
  group "memory"; begin
    let bb08 = append_block context "Bb08" fn in
    let b = builder_at_end context bb08 in

    (* RUN: grep "%build_alloca = alloca i32" < %t.ll
     * RUN: grep "%build_array_alloca = alloca i32, i32 %P2" < %t.ll
     * RUN: grep "%build_load = load i32\* %build_array_alloca" < %t.ll
     * RUN: grep "store i32 %P2, i32\* %build_alloca" < %t.ll
     * RUN: grep "%build_gep = getelementptr i32\* %build_array_alloca, i32 %P2" < %t.ll
     * RUN: grep "%build_in_bounds_gep = getelementptr inbounds i32\* %build_array_alloca, i32 %P2" < %t.ll
     * RUN: grep "%build_struct_gep = getelementptr inbounds.*%build_alloca2, i32 0, i32 1" < %t.ll
     *)
    let alloca = build_alloca i32_type "build_alloca" b in
    let array_alloca = build_array_alloca i32_type p2 "build_array_alloca" b in
    ignore(build_load array_alloca "build_load" b);
    ignore(build_store p2 alloca b);
    ignore(build_gep array_alloca [| p2 |] "build_gep" b);
    ignore(build_in_bounds_gep array_alloca [| p2 |] "build_in_bounds_gep" b);

    let sty = struct_type context [| i32_type; i8_type |] in
    let alloca2 = build_alloca sty "build_alloca2" b in
    ignore(build_struct_gep alloca2 1 "build_struct_gep" b);

    ignore(build_unreachable b)
  end;

  group "string"; begin
    let bb09 = append_block context "Bb09" fn in
    let b = builder_at_end context bb09 in
    let p = build_alloca (pointer_type i8_type) "p" b in
    (* RUN: grep "build_global_string.*stringval" < %t.ll
     * RUN: grep "store.*build_global_string1.*p" < %t.ll
     * *)
    ignore (build_global_string "stringval" "build_global_string" b);
    let g = build_global_stringptr "stringval" "build_global_string1" b in
    ignore (build_store g p b);
    ignore(build_unreachable b);
  end;

  group "casts"; begin
    let void_ptr = pointer_type i8_type in
    
    (* RUN: grep "%build_trunc = trunc i32 %P1 to i8" < %t.ll
     * RUN: grep "%build_trunc2 = trunc i32 %P1 to i8" < %t.ll
     * RUN: grep "%build_trunc3 = trunc i32 %P1 to i8" < %t.ll
     * RUN: grep "%build_zext = zext i8 %build_trunc to i32" < %t.ll
     * RUN: grep "%build_zext2 = zext i8 %build_trunc to i32" < %t.ll
     * RUN: grep "%build_sext = sext i32 %build_zext to i64" < %t.ll
     * RUN: grep "%build_sext2 = sext i32 %build_zext to i64" < %t.ll
     * RUN: grep "%build_sext3 = sext i32 %build_zext to i64" < %t.ll
     * RUN: grep "%build_uitofp = uitofp i64 %build_sext to float" < %t.ll
     * RUN: grep "%build_sitofp = sitofp i32 %build_zext to double" < %t.ll
     * RUN: grep "%build_fptoui = fptoui float %build_uitofp to i32" < %t.ll
     * RUN: grep "%build_fptosi = fptosi double %build_sitofp to i64" < %t.ll
     * RUN: grep "%build_fptrunc = fptrunc double %build_sitofp to float" < %t.ll
     * RUN: grep "%build_fptrunc2 = fptrunc double %build_sitofp to float" < %t.ll
     * RUN: grep "%build_fpext = fpext float %build_fptrunc to double" < %t.ll
     * RUN: grep "%build_fpext2 = fpext float %build_fptrunc to double" < %t.ll
     * RUN: grep "%build_inttoptr = inttoptr i32 %P1 to i8\*" < %t.ll
     * RUN: grep "%build_ptrtoint = ptrtoint i8\* %build_inttoptr to i64" < %t.ll
     * RUN: grep "%build_ptrtoint2 = ptrtoint i8\* %build_inttoptr to i64" < %t.ll
     * RUN: grep "%build_bitcast = bitcast i64 %build_ptrtoint to double" < %t.ll
     * RUN: grep "%build_bitcast2 = bitcast i64 %build_ptrtoint to double" < %t.ll
     * RUN: grep "%build_bitcast3 = bitcast i64 %build_ptrtoint to double" < %t.ll
     * RUN: grep "%build_bitcast4 = bitcast i64 %build_ptrtoint to double" < %t.ll
     * RUN: grep "%build_pointercast = bitcast i8\* %build_inttoptr to i16*" < %t.ll
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
    (* RUN: grep "%build_icmp_ne = icmp ne i32 %P1, %P2" < %t.ll
     * RUN: grep "%build_icmp_sle = icmp sle i32 %P2, %P1" < %t.ll
     * RUN: grep "%build_fcmp_false = fcmp false float %F1, %F2" < %t.ll
     * RUN: grep "%build_fcmp_true = fcmp true float %F2, %F1" < %t.ll
     * RUN: grep "%build_is_null.*= icmp eq.*%X0,.*null" < %t.ll
     * RUN: grep "%build_is_not_null = icmp ne i8\* %X1, null" < %t.ll
     * RUN: grep "%build_ptrdiff" < %t.ll
     *)
    ignore (build_icmp Icmp.Ne    p1 p2 "build_icmp_ne" atentry);
    ignore (build_icmp Icmp.Sle   p2 p1 "build_icmp_sle" atentry);
    ignore (build_fcmp Fcmp.False f1 f2 "build_fcmp_false" atentry);
    ignore (build_fcmp Fcmp.True  f2 f1 "build_fcmp_true" atentry);
    let g0 = declare_global (pointer_type i8_type) "g0" m in
    let g1 = declare_global (pointer_type i8_type) "g1" m in
    let p0 = build_load g0 "X0" atentry in
    let p1 = build_load g1 "X1" atentry in
    ignore (build_is_null p0 "build_is_null" atentry);
    ignore (build_is_not_null p1 "build_is_not_null" atentry);
    ignore (build_ptrdiff p1 p0 "build_ptrdiff" atentry);
  end;
  
  group "miscellaneous"; begin
    (* RUN: grep "%build_call = tail call cc63 i32 @.*(i32 signext %P2, i32 %P1)" < %t.ll
     * RUN: grep "%build_select = select i1 %build_icmp, i32 %P1, i32 %P2" < %t.ll
     * RUN: grep "%build_va_arg = va_arg i8\*\* null, i32" < %t.ll
     * RUN: grep "%build_extractelement = extractelement <4 x i32> %Vec1, i32 %P2" < %t.ll
     * RUN: grep "%build_insertelement = insertelement <4 x i32> %Vec1, i32 %P1, i32 %P2" < %t.ll
     * RUN: grep "%build_shufflevector = shufflevector <4 x i32> %Vec1, <4 x i32> %Vec2, <4 x i32> <i32 1, i32 1, i32 0, i32 0>" < %t.ll
     * RUN: grep "%build_insertvalue0 = insertvalue.*%bl, i32 1, 0" < %t.ll
     * RUN: grep "%build_extractvalue = extractvalue.*%build_insertvalue1, 1" < %t.ll
     *)
    let ci = build_call fn [| p2; p1 |] "build_call" atentry in
    insist (CallConv.c = instruction_call_conv ci);
    set_instruction_call_conv 63 ci;
    insist (63 = instruction_call_conv ci);
    insist (not (is_tail_call ci));
    set_tail_call true ci;
    insist (is_tail_call ci);
    add_instruction_param_attr ci 1 Attribute.Sext;
    add_instruction_param_attr ci 2 Attribute.Noalias;
    remove_instruction_param_attr ci 2 Attribute.Noalias;
    
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
    (* RUN: grep '%metadata = add i32 %P1, %P2, !test !0' < %t.ll
     * RUN: grep '!0 = metadata !{i32 1, metadata !"metadata test"}' < %t.ll
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

    clear_metadata i kind;

    insist ((has_metadata i) = false);
    insist ((metadata i kind) = None);

    set_metadata i kind md
  end;

  group "dbg"; begin
    (* RUN: grep '%dbg = add i32 %P1, %P2, !dbg !1' < %t.ll
     * RUN: grep '!1 = metadata !{i32 2, i32 3, metadata !2, metadata !2}' < %t.ll
     *)
    insist ((current_debug_location atentry) = None);

    let m_line = const_int i32_type 2 in
    let m_col = const_int i32_type 3 in
    let m_scope = mdnode context [| |] in
    let m_inlined = mdnode context [| |] in
    let md = mdnode context [| m_line; m_col; m_scope; m_inlined |] in
    set_current_debug_location atentry md;

    insist ((current_debug_location atentry) = Some md);

    let i = build_add p1 p2 "dbg" atentry in
    insist ((has_metadata i) = true);

    clear_current_debug_location atentry
  end;
  
  group "phi"; begin
    (* RUN: grep "PhiNode.*P1.*PhiBlock1.*P2.*PhiBlock2" < %t.ll
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
    
    ignore (build_unreachable at_jb);
  end


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
  suite "target"           test_target;
  suite "constants"        test_constants;
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
  suite "writer"           test_writer; (* Keep this last; it disposes m. *)
  exit !exit_status
