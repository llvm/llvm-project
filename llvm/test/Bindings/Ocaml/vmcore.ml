(* RUN: %ocamlc -warn-error A llvm.cma llvm_analysis.cma llvm_bitwriter.cma %s -o %t 2> /dev/null
 * RUN: ./%t %t.bc
 * RUN: llvm-dis < %t.bc > %t.ll
 *)

(* Note: It takes several seconds for ocamlc to link an executable with
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
let m = create_module filename
let mp = ModuleProvider.create m


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

(*===-- Types -------------------------------------------------------------===*)

let test_types () =
  (* RUN: grep {Ty01.*void} < %t.ll
   *)
  group "void";
  insist (define_type_name "Ty01" void_type m);
  insist (TypeKind.Void == classify_type void_type);

  (* RUN: grep {Ty02.*i1} < %t.ll
   *)
  group "i1";
  insist (define_type_name "Ty02" i1_type m);
  insist (TypeKind.Integer == classify_type i1_type);

  (* RUN: grep {Ty03.*i32} < %t.ll
   *)
  group "i32";
  insist (define_type_name "Ty03" i32_type m);

  (* RUN: grep {Ty04.*i42} < %t.ll
   *)
  group "i42";
  let ty = integer_type 42 in
  insist (define_type_name "Ty04" ty m);

  (* RUN: grep {Ty05.*float} < %t.ll
   *)
  group "float";
  insist (define_type_name "Ty05" float_type m);
  insist (TypeKind.Float == classify_type float_type);

  (* RUN: grep {Ty06.*double} < %t.ll
   *)
  group "double";
  insist (define_type_name "Ty06" double_type m);
  insist (TypeKind.Double == classify_type double_type);

  (* RUN: grep {Ty07.*i32.*i1, double} < %t.ll
   *)
  group "function";
  let ty = function_type i32_type [| i1_type; double_type |] in
  insist (define_type_name "Ty07" ty m);
  insist (TypeKind.Function = classify_type ty);
  insist (not (is_var_arg ty));
  insist (i32_type == return_type ty);
  insist (double_type == (param_types ty).(1));
  
  (* RUN: grep {Ty08.*\.\.\.} < %t.ll
   *)
  group "var arg function";
  let ty = var_arg_function_type void_type [| i32_type |] in
  insist (define_type_name "Ty08" ty m);
  insist (is_var_arg ty);
  
  (* RUN: grep {Ty09.*\\\[7 x i8\\\]} < %t.ll
   *)
  group "array";
  let ty = array_type i8_type 7 in
  insist (define_type_name "Ty09" ty m);
  insist (7 = array_length ty);
  insist (i8_type == element_type ty);
  insist (TypeKind.Array == classify_type ty);
  
  begin group "pointer";
    (* RUN: grep {UnqualPtrTy.*float\*} < %t.ll
     *)
    let ty = pointer_type float_type in
    insist (define_type_name "UnqualPtrTy" ty m);
    insist (float_type == element_type ty);
    insist (0 == address_space ty);
    insist (TypeKind.Pointer == classify_type ty)
  end;
  
  begin group "qualified_pointer";
    (* RUN: grep {QualPtrTy.*i8.*3.*\*} < %t.ll
     *)
    let ty = qualified_pointer_type i8_type 3 in
    insist (define_type_name "QualPtrTy" ty m);
    insist (i8_type == element_type ty);
    insist (3 == address_space ty)
  end;
  
  (* RUN: grep {Ty11.*\<4 x i16\>} < %t.ll
   *)
  group "vector";
  let ty = vector_type i16_type 4 in
  insist (define_type_name "Ty11" ty m);
  insist (i16_type == element_type ty);
  insist (4 = vector_size ty);
  
  (* RUN: grep {Ty12.*opaque} < %t.ll
   *)
  group "opaque";
  let ty = opaque_type () in
  insist (define_type_name "Ty12" ty m);
  insist (ty == ty);
  insist (ty <> opaque_type ());
  
  (* RUN: grep -v {Ty13} < %t.ll
   *)
  group "delete";
  let ty = opaque_type () in
  insist (define_type_name "Ty13" ty m);
  delete_type_name "Ty13" m;
  
  (* RUN: grep -v {RecursiveTy.*RecursiveTy} < %t.ll
   *)
  group "recursive";
  let ty = opaque_type () in
  let th = handle_to_type ty in
  refine_type ty (pointer_type ty);
  let ty = type_of_handle th in
  insist (define_type_name "RecursiveTy" ty m);
  insist (ty == element_type ty)


(*===-- Constants ---------------------------------------------------------===*)

let test_constants () =
  (* RUN: grep {Const01.*i32.*-1} < %t.ll
   *)
  group "int";
  let c = const_int i32_type (-1) in
  ignore (define_global "Const01" c m);
  insist (i32_type = type_of c);
  insist (is_constant c);

  (* RUN: grep {Const02.*i64.*-1} < %t.ll
   *)
  group "sext int";
  let c = const_int i64_type (-1) in
  ignore (define_global "Const02" c m);
  insist (i64_type = type_of c);

  (* RUN: grep {Const03.*i64.*4294967295} < %t.ll
   *)
  group "zext int64";
  let c = const_of_int64 i64_type (Int64.of_string "4294967295") false in
  ignore (define_global "Const03" c m);
  insist (i64_type = type_of c);

  (* RUN: grep {Const04.*"cruel\\\\00world"} < %t.ll
   *)
  group "string";
  let c = const_string "cruel\000world" in
  ignore (define_global "Const04" c m);
  insist ((array_type i8_type 11) = type_of c);

  (* RUN: grep {Const05.*"hi\\\\00again\\\\00"} < %t.ll
   *)
  group "stringz";
  let c = const_stringz "hi\000again" in
  ignore (define_global "Const05" c m);
  insist ((array_type i8_type 9) = type_of c);

  (* RUN: grep {ConstSingle.*2.75} < %t.ll
   * RUN: grep {ConstDouble.*3.1459} < %t.ll
   *)
  begin group "real";
    let cs = const_float float_type 2.75 in
    ignore (define_global "ConstSingle" cs m);
    insist (float_type = type_of cs);
    
    let cd = const_float double_type 3.1459 in
    ignore (define_global "ConstDouble" cd m);
    insist (double_type = type_of cd)
  end;
  
  let one = const_int i16_type 1 in
  let two = const_int i16_type 2 in
  let three = const_int i32_type 3 in
  let four = const_int i32_type 4 in
  
  (* RUN: grep {Const07.*\\\[ i32 3, i32 4 \\\]} < %t.ll
   *)
  group "array";
  let c = const_array i32_type [| three; four |] in
  ignore (define_global "Const07" c m);
  insist ((array_type i32_type 2) = (type_of c));
  
  (* RUN: grep {Const08.*< i16 1, i16 2.* >} < %t.ll
   *)
  group "vector";
  let c = const_vector [| one; two; one; two;
                          one; two; one; two |] in
  ignore (define_global "Const08" c m);
  insist ((vector_type i16_type 8) = (type_of c));
  
  (* RUN: grep {Const09.*. i16, i16, i32, i32 . .} < %t.ll
   *)
  group "structure";
  let c = const_struct [| one; two; three; four |] in
  ignore (define_global "Const09" c m);
  insist ((struct_type [| i16_type; i16_type; i32_type; i32_type |])
        = (type_of c));
  
  (* RUN: grep {Const10.*zeroinit} < %t.ll
   *)
  group "null";
  let c = const_null (packed_struct_type [| i1_type; i8_type;
                                            i64_type; double_type |]) in
  ignore (define_global "Const10" c m);
  
  (* RUN: grep {Const11.*-1} < %t.ll
   *)
  group "all ones";
  let c = const_all_ones i64_type in
  ignore (define_global "Const11" c m);
  
  (* RUN: grep {Const12.*undef} < %t.ll
   *)
  group "undef";
  let c = undef i1_type in
  ignore (define_global "Const12" c m);
  insist (i1_type = type_of c);
  insist (is_undef c);
  
  group "constant arithmetic";
  (* RUN: grep {ConstNeg.*sub} < %t.ll
   * RUN: grep {ConstNot.*xor} < %t.ll
   * RUN: grep {ConstAdd.*add} < %t.ll
   * RUN: grep {ConstSub.*sub} < %t.ll
   * RUN: grep {ConstMul.*mul} < %t.ll
   * RUN: grep {ConstUDiv.*udiv} < %t.ll
   * RUN: grep {ConstSDiv.*sdiv} < %t.ll
   * RUN: grep {ConstFDiv.*fdiv} < %t.ll
   * RUN: grep {ConstURem.*urem} < %t.ll
   * RUN: grep {ConstSRem.*srem} < %t.ll
   * RUN: grep {ConstFRem.*frem} < %t.ll
   * RUN: grep {ConstAnd.*and} < %t.ll
   * RUN: grep {ConstOr.*or} < %t.ll
   * RUN: grep {ConstXor.*xor} < %t.ll
   * RUN: grep {ConstICmp.*icmp} < %t.ll
   * RUN: grep {ConstFCmp.*fcmp} < %t.ll
   *)
  let void_ptr = pointer_type i8_type in
  let five = const_int i64_type 5 in
  let ffive = const_uitofp five double_type in
  let foldbomb_gv = define_global "FoldBomb" (const_null i8_type) m in
  let foldbomb = const_ptrtoint foldbomb_gv i64_type in
  let ffoldbomb = const_uitofp foldbomb double_type in
  ignore (define_global "ConstNeg" (const_neg foldbomb) m);
  ignore (define_global "ConstNot" (const_not foldbomb) m);
  ignore (define_global "ConstAdd" (const_add foldbomb five) m);
  ignore (define_global "ConstSub" (const_sub foldbomb five) m);
  ignore (define_global "ConstMul" (const_mul foldbomb five) m);
  ignore (define_global "ConstUDiv" (const_udiv foldbomb five) m);
  ignore (define_global "ConstSDiv" (const_sdiv foldbomb five) m);
  ignore (define_global "ConstFDiv" (const_fdiv ffoldbomb ffive) m);
  ignore (define_global "ConstURem" (const_urem foldbomb five) m);
  ignore (define_global "ConstSRem" (const_srem foldbomb five) m);
  ignore (define_global "ConstFRem" (const_frem ffoldbomb ffive) m);
  ignore (define_global "ConstAnd" (const_and foldbomb five) m);
  ignore (define_global "ConstOr" (const_or foldbomb five) m);
  ignore (define_global "ConstXor" (const_xor foldbomb five) m);
  ignore (define_global "ConstICmp" (const_icmp Icmp.Sle foldbomb five) m);
  ignore (define_global "ConstFCmp" (const_fcmp Fcmp.Ole ffoldbomb ffive) m);
  
  group "constant casts";
  (* RUN: grep {ConstTrunc.*trunc} < %t.ll
   * RUN: grep {ConstSExt.*sext} < %t.ll
   * RUN: grep {ConstZExt.*zext} < %t.ll
   * RUN: grep {ConstFPTrunc.*fptrunc} < %t.ll
   * RUN: grep {ConstFPExt.*fpext} < %t.ll
   * RUN: grep {ConstUIToFP.*uitofp} < %t.ll
   * RUN: grep {ConstSIToFP.*sitofp} < %t.ll
   * RUN: grep {ConstFPToUI.*fptoui} < %t.ll
   * RUN: grep {ConstFPToSI.*fptosi} < %t.ll
   * RUN: grep {ConstPtrToInt.*ptrtoint} < %t.ll
   * RUN: grep {ConstIntToPtr.*inttoptr} < %t.ll
   * RUN: grep {ConstBitCast.*bitcast} < %t.ll
   *)
  let i128_type = integer_type 128 in
  ignore (define_global "ConstTrunc" (const_trunc (const_add foldbomb five)
                                               i8_type) m);
  ignore (define_global "ConstSExt" (const_sext foldbomb i128_type) m);
  ignore (define_global "ConstZExt" (const_zext foldbomb i128_type) m);
  ignore (define_global "ConstFPTrunc" (const_fptrunc ffoldbomb float_type) m);
  ignore (define_global "ConstFPExt" (const_fpext ffoldbomb fp128_type) m);
  ignore (define_global "ConstUIToFP" (const_uitofp foldbomb double_type) m);
  ignore (define_global "ConstSIToFP" (const_sitofp foldbomb double_type) m);
  ignore (define_global "ConstFPToUI" (const_fptoui ffoldbomb i32_type) m);
  ignore (define_global "ConstFPToSI" (const_fptosi ffoldbomb i32_type) m);
  ignore (define_global "ConstPtrToInt" (const_ptrtoint 
    (const_gep (const_null (pointer_type i8_type))
               [| const_int i32_type 1 |])
    i32_type) m);
  ignore (define_global "ConstIntToPtr" (const_inttoptr (const_add foldbomb five)
                                                  void_ptr) m);
  ignore (define_global "ConstBitCast" (const_bitcast ffoldbomb i64_type) m);
  
  group "misc constants";
  (* RUN: grep {ConstSizeOf.*getelementptr.*null} < %t.ll
   * RUN: grep {ConstGEP.*getelementptr} < %t.ll
   * RUN: grep {ConstSelect.*select} < %t.ll
   * RUN: grep {ConstExtractElement.*extractelement} < %t.ll
   * RUN: grep {ConstInsertElement.*insertelement} < %t.ll
   * RUN: grep {ConstShuffleVector.*shufflevector} < %t.ll
   *)
  ignore (define_global "ConstSizeOf" (size_of (pointer_type i8_type)) m);
  ignore (define_global "ConstGEP" (const_gep foldbomb_gv [| five |]) m);
  ignore (define_global "ConstSelect" (const_select
    (const_icmp Icmp.Sle foldbomb five)
    (const_int i8_type (-1))
    (const_int i8_type 0)) m);
  let zero = const_int i32_type 0 in
  let one  = const_int i32_type 1 in
  ignore (define_global "ConstExtractElement" (const_extractelement
    (const_vector [| zero; one; zero; one |])
    (const_trunc foldbomb i32_type)) m);
  ignore (define_global "ConstInsertElement" (const_insertelement
    (const_vector [| zero; one; zero; one |])
    zero (const_trunc foldbomb i32_type)) m);
  ignore (define_global "ConstShuffleVector" (const_shufflevector
    (const_vector [| zero; one |])
    (const_vector [| one; zero |])
    (const_bitcast foldbomb (vector_type i32_type 2))) m)


(*===-- Global Values -----------------------------------------------------===*)

let test_global_values () =
  let (++) x f = f x; x in
  let zero32 = const_null i32_type in

  (* RUN: grep {GVal01} < %t.ll
   *)
  group "naming";
  let g = define_global "TEMPORARY" zero32 m in
  insist ("TEMPORARY" = value_name g);
  set_value_name "GVal01" g;
  insist ("GVal01" = value_name g);

  (* RUN: grep {GVal02.*linkonce} < %t.ll
   *)
  group "linkage";
  let g = define_global "GVal02" zero32 m ++
          set_linkage Linkage.Link_once in
  insist (Linkage.Link_once = linkage g);

  (* RUN: grep {GVal03.*Hanalei} < %t.ll
   *)
  group "section";
  let g = define_global "GVal03" zero32 m ++
          set_section "Hanalei" in
  insist ("Hanalei" = section g);
  
  (* RUN: grep {GVal04.*hidden} < %t.ll
   *)
  group "visibility";
  let g = define_global "GVal04" zero32 m ++
          set_visibility Visibility.Hidden in
  insist (Visibility.Hidden = visibility g);
  
  (* RUN: grep {GVal05.*align 128} < %t.ll
   *)
  group "alignment";
  let g = define_global "GVal05" zero32 m ++
          set_alignment 128 in
  insist (128 = alignment g)


(*===-- Global Variables --------------------------------------------------===*)

let test_global_variables () =
  let (++) x f = f x; x in
  let fourty_two32 = const_int i32_type 42 in

  (* RUN: grep {GVar01.*external} < %t.ll
   *)
  group "declarations";
  insist (None == lookup_global "GVar01" m);
  let g = declare_global i32_type "GVar01" m in
  insist (is_declaration g);
  insist (pointer_type float_type ==
            type_of (declare_global float_type "GVar01" m));
  insist (g == declare_global i32_type "GVar01" m);
  insist (match lookup_global "GVar01" m with Some x -> x = g
                                            | None -> false);
  
  (* RUN: grep {GVar02.*42} < %t.ll
   * RUN: grep {GVar03.*42} < %t.ll
   *)
  group "definitions";
  let g = define_global "GVar02" fourty_two32 m in
  let g2 = declare_global i32_type "GVar03" m ++
           set_initializer fourty_two32 in
  insist (not (is_declaration g));
  insist (not (is_declaration g2));
  insist ((global_initializer g) == (global_initializer g2));

  (* RUN: grep {GVar04.*thread_local} < %t.ll
   *)
  group "threadlocal";
  let g = define_global "GVar04" fourty_two32 m ++
          set_thread_local true in
  insist (is_thread_local g);

  (* RUN: grep -v {GVar05} < %t.ll
   *)
  group "delete";
  let g = define_global "GVar05" fourty_two32 m in
  delete_global g;

  (* RUN: grep -v {ConstGlobalVar.*constant} < %t.ll
   *)
  group "constant";
  let g = define_global "ConstGlobalVar" fourty_two32 m in
  insist (not (is_global_constant g));
  set_global_constant true g;
  insist (is_global_constant g);
  
  begin group "iteration";
    let m = create_module "temp" in
    
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


(*===-- Functions ---------------------------------------------------------===*)

let test_functions () =
  let ty = function_type i32_type [| i32_type; i64_type |] in
  let ty2 = function_type i8_type [| i8_type; i64_type |] in
  
  (* RUN: grep {declare i32 @Fn1\(i32, i64\)} < %t.ll
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
  
  (* RUN: grep -v {Fn2} < %t.ll
   *)
  group "delete";
  let fn = declare_function "Fn2" ty m in
  delete_function fn;
  
  (* RUN: grep {define.*Fn3} < %t.ll
   *)
  group "define";
  let fn = define_function "Fn3" ty m in
  insist (not (is_declaration fn));
  insist (1 = Array.length (basic_blocks fn));
  ignore (build_unreachable (builder_at_end (entry_block fn)));
  
  (* RUN: grep {define.*Fn4.*Param1.*Param2} < %t.ll
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
  ignore (build_unreachable (builder_at_end (entry_block fn)));
  
  (* RUN: grep {fastcc.*Fn5} < %t.ll
   *)
  group "callconv";
  let fn = define_function "Fn5" ty m in
  insist (CallConv.c = function_call_conv fn);
  set_function_call_conv CallConv.fast fn;
  insist (CallConv.fast = function_call_conv fn);
  ignore (build_unreachable (builder_at_end (entry_block fn)));
  
  begin group "gc";
    (* RUN: grep {Fn6.*gc.*shadowstack} < %t.ll
     *)
    let fn = define_function "Fn6" ty m in
    insist (None = gc fn);
    set_gc (Some "ocaml") fn;
    insist (Some "ocaml" = gc fn);
    set_gc None fn;
    insist (None = gc fn);
    set_gc (Some "shadowstack") fn;
    ignore (build_unreachable (builder_at_end (entry_block fn)));
  end;
  
  begin group "iteration";
    let m = create_module "temp" in
    
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
    let m = create_module "temp" in
    
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
  
  (* RUN: grep {Bb1} < %t.ll
   *)
  group "entry";
  let fn = declare_function "X" ty m in
  let bb = append_block "Bb1" fn in
  insist (bb = entry_block fn);
  ignore (build_unreachable (builder_at_end bb));
  
  (* RUN: grep -v Bb2 < %t.ll
   *)
  group "delete";
  let fn = declare_function "X2" ty m in
  let bb = append_block "Bb2" fn in
  delete_block bb;
  
  group "insert";
  let fn = declare_function "X3" ty m in
  let bbb = append_block "b" fn in
  let bba = insert_block "a" bbb in
  insist ([| bba; bbb |] = basic_blocks fn);
  ignore (build_unreachable (builder_at_end bba));
  ignore (build_unreachable (builder_at_end bbb));
  
  (* RUN: grep Bb3 < %t.ll
   *)
  group "name/value";
  let fn = define_function "X4" ty m in
  let bb = entry_block fn in
  ignore (build_unreachable (builder_at_end bb));
  let bbv = value_of_block bb in
  set_value_name "Bb3" bbv;
  insist ("Bb3" = value_name bbv);
  
  group "casts";
  let fn = define_function "X5" ty m in
  let bb = entry_block fn in
  ignore (build_unreachable (builder_at_end bb));
  insist (bb = block_of_value (value_of_block bb));
  insist (value_is_block (value_of_block bb));
  insist (not (value_is_block (const_null i32_type)));
  
  begin group "iteration";
    let m = create_module "temp" in
    let f = declare_function "Temp" (function_type i32_type [| |]) m in
    
    insist (At_end f = block_begin f);
    insist (At_start f = block_end f);
    
    let b1 = append_block "One" f in
    let b2 = append_block "Two" f in
    
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
    let m = create_module "temp" in
    let fty = function_type void_type [| i32_type; i32_type |] in
    let f = define_function "f" fty m in
    let bb = entry_block f in
    let b = builder_at (At_end bb) in
    
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
              ignore (insertion_block (builder ()));
              false
            with Not_found ->
              true);
    
    let fty = function_type void_type [| i32_type |] in
    let fn = define_function "BuilderParent" fty m in
    let bb = entry_block fn in
    let b = builder_at_end bb in
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
    (* RUN: grep {ret void} < %t.ll
     *)
    let fty = function_type void_type [| |] in
    let fn = declare_function "X6" fty m in
    let b = builder_at_end (append_block "Bb01" fn) in
    ignore (build_ret_void b)
  end;
  
  (* The rest of the tests will use one big function. *)
  let fty = function_type i32_type [| i32_type; i32_type |] in
  let fn = define_function "X7" fty m in
  let atentry = builder_at_end (entry_block fn) in
  let p1 = param fn 0 ++ set_value_name "P1" in
  let p2 = param fn 1 ++ set_value_name "P2" in
  let f1 = build_uitofp p1 float_type "F1" atentry in
  let f2 = build_uitofp p2 float_type "F2" atentry in
  
  let bb00 = append_block "Bb00" fn in
  ignore (build_unreachable (builder_at_end bb00));
  
  group "ret"; begin
    (* RUN: grep {ret.*P1} < %t.ll
     *)
    let ret = build_ret p1 atentry in
    position_before ret atentry
  end;
  
  group "br"; begin
    (* RUN: grep {br.*Bb02} < %t.ll
     *)
    let bb02 = append_block "Bb02" fn in
    let b = builder_at_end bb02 in
    ignore (build_br bb02 b)
  end;
  
  group "cond_br"; begin
    (* RUN: grep {br.*Inst01.*Bb03.*Bb00} < %t.ll
     *)
    let bb03 = append_block "Bb03" fn in
    let b = builder_at_end bb03 in
    let cond = build_trunc p1 i1_type "Inst01" b in
    ignore (build_cond_br cond bb03 bb00 b)
  end;
  
  group "switch"; begin
    (* RUN: grep {switch.*P1.*SwiBlock3} < %t.ll
     * RUN: grep {2,.*SwiBlock2} < %t.ll
     *)
    let bb1 = append_block "SwiBlock1" fn in
    let bb2 = append_block "SwiBlock2" fn in
    ignore (build_unreachable (builder_at_end bb2));
    let bb3 = append_block "SwiBlock3" fn in
    ignore (build_unreachable (builder_at_end bb3));
    let si = build_switch p1 bb3 1 (builder_at_end bb1) in
    ignore (add_case si (const_int i32_type 2) bb2)
  end;
  
  group "invoke"; begin
    (* RUN: grep {Inst02.*invoke.*P1.*P2} < %t.ll
     * RUN: grep {to.*Bb04.*unwind.*Bb00} < %t.ll
     *)
    let bb04 = append_block "Bb04" fn in
    let b = builder_at_end bb04 in
    ignore (build_invoke fn [| p1; p2 |] bb04 bb00 "Inst02" b)
  end;
  
  group "unwind"; begin
    (* RUN: grep {unwind} < %t.ll
     *)
    let bb05 = append_block "Bb05" fn in
    let b = builder_at_end bb05 in
    ignore (build_unwind b)
  end;
  
  group "unreachable"; begin
    (* RUN: grep {unreachable} < %t.ll
     *)
    let bb06 = append_block "Bb06" fn in
    let b = builder_at_end bb06 in
    ignore (build_unreachable b)
  end;
  
  group "arithmetic"; begin
    let bb07 = append_block "Bb07" fn in
    let b = builder_at_end bb07 in
    
    (* RUN: grep {Inst03.*add.*P1.*P2} < %t.ll
     * RUN: grep {Inst04.*sub.*P1.*Inst03} < %t.ll
     * RUN: grep {Inst05.*mul.*P1.*Inst04} < %t.ll
     * RUN: grep {Inst06.*udiv.*P1.*Inst05} < %t.ll
     * RUN: grep {Inst07.*sdiv.*P1.*Inst06} < %t.ll
     * RUN: grep {Inst08.*fdiv.*F1.*F2} < %t.ll
     * RUN: grep {Inst09.*urem.*P1.*Inst07} < %t.ll
     * RUN: grep {Inst10.*srem.*P1.*Inst09} < %t.ll
     * RUN: grep {Inst11.*frem.*F1.*Inst08} < %t.ll
     * RUN: grep {Inst12.*shl.*P1.*Inst10} < %t.ll
     * RUN: grep {Inst13.*lshr.*P1.*Inst12} < %t.ll
     * RUN: grep {Inst14.*ashr.*P1.*Inst13} < %t.ll
     * RUN: grep {Inst15.*and.*P1.*Inst14} < %t.ll
     * RUN: grep {Inst16.*or.*P1.*Inst15} < %t.ll
     * RUN: grep {Inst17.*xor.*P1.*Inst16} < %t.ll
     * RUN: grep {Inst18.*sub.*0.*Inst17} < %t.ll
     * RUN: grep {Inst19.*xor.*Inst18.*-1} < %t.ll
     *)
    let inst03 = build_add  p1 p2     "Inst03" b in
    let inst04 = build_sub  p1 inst03 "Inst04" b in
    let inst05 = build_mul  p1 inst04 "Inst05" b in
    let inst06 = build_udiv p1 inst05 "Inst06" b in
    let inst07 = build_sdiv p1 inst06 "Inst07" b in
    let inst08 = build_fdiv f1 f2     "Inst08" b in
    let inst09 = build_urem p1 inst07 "Inst09" b in
    let inst10 = build_srem p1 inst09 "Inst10" b in
          ignore(build_frem f1 inst08 "Inst11" b);
    let inst12 = build_shl  p1 inst10 "Inst12" b in
    let inst13 = build_lshr p1 inst12 "Inst13" b in
    let inst14 = build_ashr p1 inst13 "Inst14" b in
    let inst15 = build_and  p1 inst14 "Inst15" b in
    let inst16 = build_or   p1 inst15 "Inst16" b in
    let inst17 = build_xor  p1 inst16 "Inst17" b in
    let inst18 = build_neg  inst17    "Inst18" b in
         ignore (build_not  inst18    "Inst19" b);
         ignore (build_unreachable b)
  end;
  
  group "memory"; begin
    let bb08 = append_block "Bb08" fn in
    let b = builder_at_end bb08 in
    
    (* RUN: grep {Inst20.*malloc.*i8	} < %t.ll
     * RUN: grep {Inst21.*malloc.*i8.*P1} < %t.ll
     * RUN: grep {Inst22.*alloca.*i32	} < %t.ll
     * RUN: grep {Inst23.*alloca.*i32.*P2} < %t.ll
     * RUN: grep {free.*Inst20} < %t.ll
     * RUN: grep {Inst25.*load.*Inst21} < %t.ll
     * RUN: grep {store.*P2.*Inst22} < %t.ll
     * RUN: grep {Inst27.*getelementptr.*Inst23.*P2} < %t.ll
     *)
    let inst20 = build_malloc i8_type "Inst20" b in
    let inst21 = build_array_malloc i8_type p1 "Inst21" b in
    let inst22 = build_alloca i32_type "Inst22" b in
    let inst23 = build_array_alloca i32_type p2 "Inst23" b in
          ignore(build_free inst20 b);
          ignore(build_load inst21 "Inst25" b);
          ignore(build_store p2 inst22 b);
          ignore(build_gep inst23 [| p2 |] "Inst27" b);
          ignore(build_unreachable b)
  end;
  
  group "casts"; begin
    let void_ptr = pointer_type i8_type in
    
    (* RUN: grep {Inst28.*trunc.*P1.*i8} < %t.ll
     * RUN: grep {Inst29.*zext.*Inst28.*i32} < %t.ll
     * RUN: grep {Inst30.*sext.*Inst29.*i64} < %t.ll
     * RUN: grep {Inst31.*uitofp.*Inst30.*float} < %t.ll
     * RUN: grep {Inst32.*sitofp.*Inst29.*double} < %t.ll
     * RUN: grep {Inst33.*fptoui.*Inst31.*i32} < %t.ll
     * RUN: grep {Inst34.*fptosi.*Inst32.*i64} < %t.ll
     * RUN: grep {Inst35.*fptrunc.*Inst32.*float} < %t.ll
     * RUN: grep {Inst36.*fpext.*Inst35.*double} < %t.ll
     * RUN: grep {Inst37.*inttoptr.*P1.*i8\*} < %t.ll
     * RUN: grep {Inst38.*ptrtoint.*Inst37.*i64} < %t.ll
     * RUN: grep {Inst39.*bitcast.*Inst38.*double} < %t.ll
     *)
    let inst28 = build_trunc p1 i8_type "Inst28" atentry in
    let inst29 = build_zext inst28 i32_type "Inst29" atentry in
    let inst30 = build_sext inst29 i64_type "Inst30" atentry in
    let inst31 = build_uitofp inst30 float_type "Inst31" atentry in
    let inst32 = build_sitofp inst29 double_type "Inst32" atentry in
          ignore(build_fptoui inst31 i32_type "Inst33" atentry);
          ignore(build_fptosi inst32 i64_type "Inst34" atentry);
    let inst35 = build_fptrunc inst32 float_type "Inst35" atentry in
          ignore(build_fpext inst35 double_type "Inst36" atentry);
    let inst37 = build_inttoptr p1 void_ptr "Inst37" atentry in
    let inst38 = build_ptrtoint inst37 i64_type "Inst38" atentry in
          ignore(build_bitcast inst38 double_type "Inst39" atentry)
  end;
  
  group "comparisons"; begin
    (* RUN: grep {Inst40.*icmp.*ne.*P1.*P2} < %t.ll
     * RUN: grep {Inst41.*icmp.*sle.*P2.*P1} < %t.ll
     * RUN: grep {Inst42.*fcmp.*false.*F1.*F2} < %t.ll
     * RUN: grep {Inst43.*fcmp.*true.*F2.*F1} < %t.ll
     *)
    ignore (build_icmp Icmp.Ne    p1 p2 "Inst40" atentry);
    ignore (build_icmp Icmp.Sle   p2 p1 "Inst41" atentry);
    ignore (build_fcmp Fcmp.False f1 f2 "Inst42" atentry);
    ignore (build_fcmp Fcmp.True  f2 f1 "Inst43" atentry)
  end;
  
  group "miscellaneous"; begin
    (* RUN: grep {CallInst.*call.*P2.*P1} < %t.ll
     * RUN: grep {CallInst.*cc63} < %t.ll
     * RUN: grep {Inst47.*select.*Inst46.*P1.*P2} < %t.ll
     * RUN: grep {Inst48.*va_arg.*null.*i32} < %t.ll
     * RUN: grep {Inst49.*extractelement.*Vec1.*P2} < %t.ll
     * RUN: grep {Inst50.*insertelement.*Vec1.*P1.*P2} < %t.ll
     * RUN: grep {Inst51.*shufflevector.*Vec1.*Vec2.*1.*1.*0.*0} < %t.ll
     * RUN: grep {CallInst.*tail call} < %t.ll
     *)
    let ci = build_call fn [| p2; p1 |] "CallInst" atentry in
    insist (CallConv.c = instruction_call_conv ci);
    set_instruction_call_conv 63 ci;
    insist (63 = instruction_call_conv ci);
    insist (not (is_tail_call ci));
    set_tail_call true ci;
    insist (is_tail_call ci);
    
    let inst46 = build_icmp Icmp.Eq p1 p2 "Inst46" atentry in
         ignore (build_select inst46 p1 p2 "Inst47" atentry);
         ignore (build_va_arg
                  (const_null (pointer_type (pointer_type i8_type)))
                  i32_type "Inst48" atentry);
    
    (* Set up some vector vregs. *)
    let one  = const_int i32_type 1 in
    let zero = const_int i32_type 0 in
    let t1 = const_vector [| one; zero; one; zero |] in
    let t2 = const_vector [| zero; one; zero; one |] in
    let t3 = const_vector [| one; one; zero; zero |] in
    let vec1 = build_insertelement t1 p1 p2 "Vec1" atentry in
    let vec2 = build_insertelement t2 p1 p2 "Vec2" atentry in
    
    ignore (build_extractelement vec1 p2 "Inst49" atentry);
    ignore (build_insertelement vec1 p1 p2 "Inst50" atentry);
    ignore (build_shufflevector vec1 vec2 t3 "Inst51" atentry);
  end;
  
  group "phi"; begin
    (* RUN: grep {PhiNode.*P1.*PhiBlock1.*P2.*PhiBlock2} < %t.ll
     *)
    let b1 = append_block "PhiBlock1" fn in
    let b2 = append_block "PhiBlock2" fn in
    
    let jb = append_block "PhiJoinBlock" fn in
    ignore (build_br jb (builder_at_end b1));
    ignore (build_br jb (builder_at_end b2));
    let at_jb = builder_at_end jb in
    
    let phi = build_phi [(p1, b1)] "PhiNode" at_jb in
    insist ([(p1, b1)] = incoming phi);
    
    add_incoming (p2, b2) phi;
    insist ([(p1, b1); (p2, b2)] = incoming phi);
    
    ignore (build_unreachable at_jb);
  end


(*===-- Module Provider ---------------------------------------------------===*)

let test_module_provider () =
  let m = create_module "test" in
  let mp = ModuleProvider.create m in
  ModuleProvider.dispose mp


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
    ignore (build_ret_void (builder_at_end (entry_block fn)));
    
    ignore (PassManager.create_function mp
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
  
  ModuleProvider.dispose mp


(*===-- Driver ------------------------------------------------------------===*)

let _ =
  suite "target"           test_target;
  suite "types"            test_types;
  suite "constants"        test_constants;
  suite "global values"    test_global_values;
  suite "global variables" test_global_variables;
  suite "functions"        test_functions;
  suite "params"           test_params;
  suite "basic blocks"     test_basic_blocks;
  suite "instructions"     test_instructions;
  suite "builder"          test_builder;
  suite "module provider"  test_module_provider;
  suite "pass manager"     test_pass_manager;
  suite "writer"           test_writer; (* Keep this last; it disposes m. *)
  exit !exit_status
