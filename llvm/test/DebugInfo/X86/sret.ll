; RUN: llc -split-dwarf=Enable -O0 %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t
; RUN: llvm-dwarfdump -debug-dump=all %t | FileCheck %s

; Based on the debuginfo-tests/sret.cpp code.

; CHECK: DW_AT_GNU_dwo_id [DW_FORM_data8] (0x51ac5644b1937aa1)
; CHECK: DW_AT_GNU_dwo_id [DW_FORM_data8] (0x51ac5644b1937aa1)

%class.A = type { i32 (...)**, i32 }
%class.B = type { i8 }

@_ZTV1A = linkonce_odr unnamed_addr constant [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI1A to i8*), i8* bitcast (void (%class.A*)* @_ZN1AD2Ev to i8*), i8* bitcast (void (%class.A*)* @_ZN1AD0Ev to i8*)]
@_ZTVN10__cxxabiv117__class_type_infoE = external global i8*
@_ZTS1A = linkonce_odr constant [3 x i8] c"1A\00"
@_ZTI1A = linkonce_odr constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([3 x i8]* @_ZTS1A, i32 0, i32 0) }

@_ZN1AC1Ei = alias void (%class.A*, i32)* @_ZN1AC2Ei
@_ZN1AC1ERKS_ = alias void (%class.A*, %class.A*)* @_ZN1AC2ERKS_

; Function Attrs: nounwind uwtable
define void @_ZN1AC2Ei(%class.A* %this, i32 %i) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  %i.addr = alloca i32, align 4
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !67, metadata !{!"0x102"}), !dbg !69
  store i32 %i, i32* %i.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %i.addr, metadata !70, metadata !{!"0x102"}), !dbg !71
  %this1 = load %class.A** %this.addr
  %0 = bitcast %class.A* %this1 to i8***, !dbg !72
  store i8** getelementptr inbounds ([4 x i8*]* @_ZTV1A, i64 0, i64 2), i8*** %0, !dbg !72
  %m_int = getelementptr inbounds %class.A* %this1, i32 0, i32 1, !dbg !72
  %1 = load i32* %i.addr, align 4, !dbg !72
  store i32 %1, i32* %m_int, align 4, !dbg !72
  ret void, !dbg !73
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind uwtable
define void @_ZN1AC2ERKS_(%class.A* %this, %class.A* %rhs) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  %rhs.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !74, metadata !{!"0x102"}), !dbg !75
  store %class.A* %rhs, %class.A** %rhs.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %rhs.addr, metadata !76, metadata !{!"0x102"}), !dbg !77
  %this1 = load %class.A** %this.addr
  %0 = bitcast %class.A* %this1 to i8***, !dbg !78
  store i8** getelementptr inbounds ([4 x i8*]* @_ZTV1A, i64 0, i64 2), i8*** %0, !dbg !78
  %m_int = getelementptr inbounds %class.A* %this1, i32 0, i32 1, !dbg !78
  %1 = load %class.A** %rhs.addr, align 8, !dbg !78
  %m_int2 = getelementptr inbounds %class.A* %1, i32 0, i32 1, !dbg !78
  %2 = load i32* %m_int2, align 4, !dbg !78
  store i32 %2, i32* %m_int, align 4, !dbg !78
  ret void, !dbg !79
}

; Function Attrs: nounwind uwtable
define %class.A* @_ZN1AaSERKS_(%class.A* %this, %class.A* %rhs) #0 align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  %rhs.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !80, metadata !{!"0x102"}), !dbg !81
  store %class.A* %rhs, %class.A** %rhs.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %rhs.addr, metadata !82, metadata !{!"0x102"}), !dbg !83
  %this1 = load %class.A** %this.addr
  %0 = load %class.A** %rhs.addr, align 8, !dbg !84
  %m_int = getelementptr inbounds %class.A* %0, i32 0, i32 1, !dbg !84
  %1 = load i32* %m_int, align 4, !dbg !84
  %m_int2 = getelementptr inbounds %class.A* %this1, i32 0, i32 1, !dbg !84
  store i32 %1, i32* %m_int2, align 4, !dbg !84
  ret %class.A* %this1, !dbg !85
}

; Function Attrs: nounwind uwtable
define i32 @_ZN1A7get_intEv(%class.A* %this) #0 align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !86, metadata !{!"0x102"}), !dbg !87
  %this1 = load %class.A** %this.addr
  %m_int = getelementptr inbounds %class.A* %this1, i32 0, i32 1, !dbg !88
  %0 = load i32* %m_int, align 4, !dbg !88
  ret i32 %0, !dbg !88
}

; Function Attrs: uwtable
define void @_ZN1B9AInstanceEv(%class.A* noalias sret %agg.result, %class.B* %this) #2 align 2 {
entry:
  %this.addr = alloca %class.B*, align 8
  %nrvo = alloca i1
  %cleanup.dest.slot = alloca i32
  store %class.B* %this, %class.B** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.B** %this.addr, metadata !89, metadata !{!"0x102"}), !dbg !91
  %this1 = load %class.B** %this.addr
  store i1 false, i1* %nrvo, !dbg !92
  call void @llvm.dbg.declare(metadata %class.A* %agg.result, metadata !93, metadata !{!"0x102"}), !dbg !92
  call void @_ZN1AC1Ei(%class.A* %agg.result, i32 12), !dbg !92
  store i1 true, i1* %nrvo, !dbg !94
  store i32 1, i32* %cleanup.dest.slot
  %nrvo.val = load i1* %nrvo, !dbg !95
  br i1 %nrvo.val, label %nrvo.skipdtor, label %nrvo.unused, !dbg !95

nrvo.unused:                                      ; preds = %entry
  call void @_ZN1AD2Ev(%class.A* %agg.result), !dbg !96
  br label %nrvo.skipdtor, !dbg !96

nrvo.skipdtor:                                    ; preds = %nrvo.unused, %entry
  ret void, !dbg !98
}

; Function Attrs: nounwind uwtable
define linkonce_odr void @_ZN1AD2Ev(%class.A* %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !101, metadata !{!"0x102"}), !dbg !102
  %this1 = load %class.A** %this.addr
  ret void, !dbg !103
}

; Function Attrs: uwtable
define i32 @main(i32 %argc, i8** %argv) #2 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  %b = alloca %class.B, align 1
  %return_val = alloca i32, align 4
  %temp.lvalue = alloca %class.A, align 8
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  %a = alloca %class.A, align 8
  %cleanup.dest.slot = alloca i32
  store i32 0, i32* %retval
  store i32 %argc, i32* %argc.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !104, metadata !{!"0x102"}), !dbg !105
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !106, metadata !{!"0x102"}), !dbg !105
  call void @llvm.dbg.declare(metadata %class.B* %b, metadata !107, metadata !{!"0x102"}), !dbg !108
  call void @_ZN1BC2Ev(%class.B* %b), !dbg !108
  call void @llvm.dbg.declare(metadata i32* %return_val, metadata !109, metadata !{!"0x102"}), !dbg !110
  call void @_ZN1B9AInstanceEv(%class.A* sret %temp.lvalue, %class.B* %b), !dbg !110
  %call = invoke i32 @_ZN1A7get_intEv(%class.A* %temp.lvalue)
          to label %invoke.cont unwind label %lpad, !dbg !110

invoke.cont:                                      ; preds = %entry
  call void @_ZN1AD2Ev(%class.A* %temp.lvalue), !dbg !111
  store i32 %call, i32* %return_val, align 4, !dbg !111
  call void @llvm.dbg.declare(metadata %class.A* %a, metadata !113, metadata !{!"0x102"}), !dbg !114
  call void @_ZN1B9AInstanceEv(%class.A* sret %a, %class.B* %b), !dbg !114
  %0 = load i32* %return_val, align 4, !dbg !115
  store i32 %0, i32* %retval, !dbg !115
  store i32 1, i32* %cleanup.dest.slot
  call void @_ZN1AD2Ev(%class.A* %a), !dbg !116
  %1 = load i32* %retval, !dbg !116
  ret i32 %1, !dbg !116

lpad:                                             ; preds = %entry
  %2 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          cleanup, !dbg !116
  %3 = extractvalue { i8*, i32 } %2, 0, !dbg !116
  store i8* %3, i8** %exn.slot, !dbg !116
  %4 = extractvalue { i8*, i32 } %2, 1, !dbg !116
  store i32 %4, i32* %ehselector.slot, !dbg !116
  invoke void @_ZN1AD2Ev(%class.A* %temp.lvalue)
          to label %invoke.cont1 unwind label %terminate.lpad, !dbg !116

invoke.cont1:                                     ; preds = %lpad
  br label %eh.resume, !dbg !117

eh.resume:                                        ; preds = %invoke.cont1
  %exn = load i8** %exn.slot, !dbg !119
  %sel = load i32* %ehselector.slot, !dbg !119
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn, 0, !dbg !119
  %lpad.val2 = insertvalue { i8*, i32 } %lpad.val, i32 %sel, 1, !dbg !119
  resume { i8*, i32 } %lpad.val2, !dbg !119

terminate.lpad:                                   ; preds = %lpad
  %5 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          catch i8* null, !dbg !121
  %6 = extractvalue { i8*, i32 } %5, 0, !dbg !121
  call void @__clang_call_terminate(i8* %6) #5, !dbg !121
  unreachable, !dbg !121
}

; Function Attrs: nounwind uwtable
define linkonce_odr void @_ZN1BC2Ev(%class.B* %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %class.B*, align 8
  store %class.B* %this, %class.B** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.B** %this.addr, metadata !123, metadata !{!"0x102"}), !dbg !124
  %this1 = load %class.B** %this.addr
  ret void, !dbg !125
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: noinline noreturn nounwind
define linkonce_odr hidden void @__clang_call_terminate(i8*) #3 {
  %2 = call i8* @__cxa_begin_catch(i8* %0) #6
  call void @_ZSt9terminatev() #5
  unreachable
}

declare i8* @__cxa_begin_catch(i8*)

declare void @_ZSt9terminatev()

; Function Attrs: uwtable
define linkonce_odr void @_ZN1AD0Ev(%class.A* %this) unnamed_addr #2 align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !126, metadata !{!"0x102"}), !dbg !127
  %this1 = load %class.A** %this.addr
  invoke void @_ZN1AD2Ev(%class.A* %this1)
          to label %invoke.cont unwind label %lpad, !dbg !128

invoke.cont:                                      ; preds = %entry
  %0 = bitcast %class.A* %this1 to i8*, !dbg !129
  call void @_ZdlPv(i8* %0) #7, !dbg !129
  ret void, !dbg !129

lpad:                                             ; preds = %entry
  %1 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          cleanup, !dbg !131
  %2 = extractvalue { i8*, i32 } %1, 0, !dbg !131
  store i8* %2, i8** %exn.slot, !dbg !131
  %3 = extractvalue { i8*, i32 } %1, 1, !dbg !131
  store i32 %3, i32* %ehselector.slot, !dbg !131
  %4 = bitcast %class.A* %this1 to i8*, !dbg !131
  call void @_ZdlPv(i8* %4) #7, !dbg !131
  br label %eh.resume, !dbg !131

eh.resume:                                        ; preds = %lpad
  %exn = load i8** %exn.slot, !dbg !133
  %sel = load i32* %ehselector.slot, !dbg !133
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn, 0, !dbg !133
  %lpad.val2 = insertvalue { i8*, i32 } %lpad.val, i32 %sel, 1, !dbg !133
  resume { i8*, i32 } %lpad.val2, !dbg !133
}

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPv(i8*) #4

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { noinline noreturn nounwind }
attributes #4 = { nobuiltin nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { noreturn nounwind }
attributes #6 = { nounwind }
attributes #7 = { builtin nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!64, !65}
!llvm.ident = !{!66}

!0 = !{!"0x11\004\00clang version 3.5.0 (trunk 203283) (llvm/trunk 203307)\000\00\000\00sret.dwo\001", !1, !2, !3, !48, !2, !2} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/tmp/sret.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"sret.cpp", !"/usr/local/google/home/echristo/tmp"}
!2 = !{}
!3 = !{!4, !37}
!4 = !{!"0x2\00A\001\00128\0064\000\000\000", !1, null, null, !5, !"_ZTS1A", null, !"_ZTS1A"} ; [ DW_TAG_class_type ] [A] [line 1, size 128, align 64, offset 0] [def] [from ]
!5 = !{!6, !13, !14, !19, !25, !29, !33}
!6 = !{!"0xd\00_vptr$A\000\0064\000\000\0064", !1, !7, !8} ; [ DW_TAG_member ] [_vptr$A] [line 0, size 64, align 0, offset 0] [artificial] [from ]
!7 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/usr/local/google/home/echristo/tmp/sret.cpp]
!8 = !{!"0xf\00\000\0064\000\000\000", null, null, !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 0, offset 0] [from __vtbl_ptr_type]
!9 = !{!"0xf\00__vtbl_ptr_type\000\0064\000\000\000", null, null, !10} ; [ DW_TAG_pointer_type ] [__vtbl_ptr_type] [line 0, size 64, align 0, offset 0] [from ]
!10 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !11, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!11 = !{!12}
!12 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!13 = !{!"0xd\00m_int\0013\0032\0032\0064\002", !1, !"_ZTS1A", !12} ; [ DW_TAG_member ] [m_int] [line 13, size 32, align 32, offset 64] [protected] [from int]
!14 = !{!"0x2e\00A\00A\00\004\000\000\000\006\00256\000\004", !1, !"_ZTS1A", !15, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ] [line 4] [A]
!15 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !16, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = !{null, !17, !12}
!17 = !{!"0xf\00\000\0064\0064\000\001088", null, null, !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1A]
!19 = !{!"0x2e\00A\00A\00\005\000\000\000\006\00256\000\005", !1, !"_ZTS1A", !20, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ] [line 5] [A]
!20 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !21, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!21 = !{null, !17, !22}
!22 = !{!"0x10\00\000\000\000\000\000", null, null, !23} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from ]
!23 = !{!"0x26\00\000\000\000\000\000", null, null, !"_ZTS1A"} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from _ZTS1A]
!25 = !{!"0x2e\00operator=\00operator=\00_ZN1AaSERKS_\007\000\000\000\006\00256\000\007", !1, !"_ZTS1A", !26, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ] [line 7] [operator=]
!26 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !27, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!27 = !{!22, !17, !22}
!29 = !{!"0x2e\00~A\00~A\00\008\000\000\001\006\00256\000\008", !1, !"_ZTS1A", !30, !"_ZTS1A", null, null, i32 0, null} ; [ DW_TAG_subprogram ] [line 8] [~A]
!30 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !31, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!31 = !{null, !17}
!33 = !{!"0x2e\00get_int\00get_int\00_ZN1A7get_intEv\0010\000\000\000\006\00256\000\0010", !1, !"_ZTS1A", !34, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ] [line 10] [get_int]
!34 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !35, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!35 = !{!12, !17}
!37 = !{!"0x2\00B\0038\008\008\000\000\000", !1, null, null, !38, null, null, !"_ZTS1B"} ; [ DW_TAG_class_type ] [B] [line 38, size 8, align 8, offset 0] [def] [from ]
!38 = !{!39, !44}
!39 = !{!"0x2e\00B\00B\00\0041\000\000\000\006\00256\000\0041", !1, !"_ZTS1B", !40, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ] [line 41] [B]
!40 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !41, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!41 = !{null, !42}
!42 = !{!"0xf\00\000\0064\0064\000\001088", null, null, !"_ZTS1B"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1B]
!44 = !{!"0x2e\00AInstance\00AInstance\00_ZN1B9AInstanceEv\0043\000\000\000\006\00256\000\0043", !1, !"_ZTS1B", !45, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ] [line 43] [AInstance]
!45 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !46, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!46 = !{!4, !42}
!48 = !{!49, !50, !51, !52, !53, !54, !61, !62, !63}
!49 = !{!"0x2e\00A\00A\00_ZN1AC2Ei\0016\000\001\000\006\00256\000\0018", !1, !"_ZTS1A", !15, null, void (%class.A*, i32)* @_ZN1AC2Ei, null, !14, !2} ; [ DW_TAG_subprogram ] [line 16] [def] [scope 18] [A]
!50 = !{!"0x2e\00A\00A\00_ZN1AC2ERKS_\0021\000\001\000\006\00256\000\0023", !1, !"_ZTS1A", !20, null, void (%class.A*, %class.A*)* @_ZN1AC2ERKS_, null, !19, !2} ; [ DW_TAG_subprogram ] [line 21] [def] [scope 23] [A]
!51 = !{!"0x2e\00operator=\00operator=\00_ZN1AaSERKS_\0027\000\001\000\006\00256\000\0028", !1, !"_ZTS1A", !26, null, %class.A* (%class.A*, %class.A*)* @_ZN1AaSERKS_, null, !25, !2} ; [ DW_TAG_subprogram ] [line 27] [def] [scope 28] [operator=]
!52 = !{!"0x2e\00get_int\00get_int\00_ZN1A7get_intEv\0033\000\001\000\006\00256\000\0034", !1, !"_ZTS1A", !34, null, i32 (%class.A*)* @_ZN1A7get_intEv, null, !33, !2} ; [ DW_TAG_subprogram ] [line 33] [def] [scope 34] [get_int]
!53 = !{!"0x2e\00AInstance\00AInstance\00_ZN1B9AInstanceEv\0047\000\001\000\006\00256\000\0048", !1, !"_ZTS1B", !45, null, void (%class.A*, %class.B*)* @_ZN1B9AInstanceEv, null, !44, !2} ; [ DW_TAG_subprogram ] [line 47] [def] [scope 48] [AInstance]
!54 = !{!"0x2e\00main\00main\00\0053\000\001\000\006\00256\000\0054", !1, !7, !55, null, i32 (i32, i8**)* @main, null, null, !2} ; [ DW_TAG_subprogram ] [line 53] [def] [scope 54] [main]
!55 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !56, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!56 = !{!12, !12, !57}
!57 = !{!"0xf\00\000\0064\0064\000\000", null, null, !58} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!58 = !{!"0xf\00\000\0064\0064\000\000", null, null, !59} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!59 = !{!"0x26\00\000\000\000\000\000", null, null, !60} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from char]
!60 = !{!"0x24\00char\000\008\008\000\000\006", null, null} ; [ DW_TAG_base_type ] [char] [line 0, size 8, align 8, offset 0, enc DW_ATE_signed_char]
!61 = !{!"0x2e\00~A\00~A\00_ZN1AD0Ev\008\000\001\000\006\00256\000\008", !1, !"_ZTS1A", !30, null, void (%class.A*)* @_ZN1AD0Ev, null, !29, !2} ; [ DW_TAG_subprogram ] [line 8] [def] [~A]
!62 = !{!"0x2e\00B\00B\00_ZN1BC2Ev\0041\000\001\000\006\00256\000\0041", !1, !"_ZTS1B", !40, null, void (%class.B*)* @_ZN1BC2Ev, null, !39, !2} ; [ DW_TAG_subprogram ] [line 41] [def] [B]
!63 = !{!"0x2e\00~A\00~A\00_ZN1AD2Ev\008\000\001\000\006\00256\000\008", !1, !"_ZTS1A", !30, null, void (%class.A*)* @_ZN1AD2Ev, null, !29, !2} ; [ DW_TAG_subprogram ] [line 8] [def] [~A]
!64 = !{i32 2, !"Dwarf Version", i32 4}
!65 = !{i32 1, !"Debug Info Version", i32 2}
!66 = !{!"clang version 3.5.0 (trunk 203283) (llvm/trunk 203307)"}
!67 = !{!"0x101\00this\0016777216\001088", !49, null, !68} ; [ DW_TAG_arg_variable ] [this] [line 0]
!68 = !{!"0xf\00\000\0064\0064\000\000", null, null, !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS1A]
!69 = !MDLocation(line: 0, scope: !49)
!70 = !{!"0x101\00i\0033554448\000", !49, !7, !12} ; [ DW_TAG_arg_variable ] [i] [line 16]
!71 = !MDLocation(line: 16, scope: !49)
!72 = !MDLocation(line: 18, scope: !49)
!73 = !MDLocation(line: 19, scope: !49)
!74 = !{!"0x101\00this\0016777216\001088", !50, null, !68} ; [ DW_TAG_arg_variable ] [this] [line 0]
!75 = !MDLocation(line: 0, scope: !50)
!76 = !{!"0x101\00rhs\0033554453\000", !50, !7, !22} ; [ DW_TAG_arg_variable ] [rhs] [line 21]
!77 = !MDLocation(line: 21, scope: !50)
!78 = !MDLocation(line: 23, scope: !50)
!79 = !MDLocation(line: 24, scope: !50)
!80 = !{!"0x101\00this\0016777216\001088", !51, null, !68} ; [ DW_TAG_arg_variable ] [this] [line 0]
!81 = !MDLocation(line: 0, scope: !51)
!82 = !{!"0x101\00rhs\0033554459\000", !51, !7, !22} ; [ DW_TAG_arg_variable ] [rhs] [line 27]
!83 = !MDLocation(line: 27, scope: !51)
!84 = !MDLocation(line: 29, scope: !51)
!85 = !MDLocation(line: 30, scope: !51)
!86 = !{!"0x101\00this\0016777216\001088", !52, null, !68} ; [ DW_TAG_arg_variable ] [this] [line 0]
!87 = !MDLocation(line: 0, scope: !52)
!88 = !MDLocation(line: 35, scope: !52)
!89 = !{!"0x101\00this\0016777216\001088", !53, null, !90} ; [ DW_TAG_arg_variable ] [this] [line 0]
!90 = !{!"0xf\00\000\0064\0064\000\000", null, null, !"_ZTS1B"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS1B]
!91 = !MDLocation(line: 0, scope: !53)
!92 = !MDLocation(line: 49, scope: !53)
!93 = !{!"0x100\00a\0049\008192", !53, !7, !4} ; [ DW_TAG_auto_variable ] [a] [line 49]
!94 = !MDLocation(line: 50, scope: !53)
!95 = !MDLocation(line: 51, scope: !53)
!96 = !MDLocation(line: 51, scope: !97)
!97 = !{!"0xb\0051\000\002", !1, !53} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/sret.cpp]
!98 = !MDLocation(line: 51, scope: !99)
!99 = !{!"0xb\0051\000\003", !1, !100} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/sret.cpp]
!100 = !{!"0xb\0051\000\001", !1, !53} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/sret.cpp]
!101 = !{!"0x101\00this\0016777216\001088", !63, null, !68} ; [ DW_TAG_arg_variable ] [this] [line 0]
!102 = !MDLocation(line: 0, scope: !63)
!103 = !MDLocation(line: 8, scope: !63)
!104 = !{!"0x101\00argc\0016777269\000", !54, !7, !12} ; [ DW_TAG_arg_variable ] [argc] [line 53]
!105 = !MDLocation(line: 53, scope: !54)
!106 = !{!"0x101\00argv\0033554485\000", !54, !7, !57} ; [ DW_TAG_arg_variable ] [argv] [line 53]
!107 = !{!"0x100\00b\0055\000", !54, !7, !37} ; [ DW_TAG_auto_variable ] [b] [line 55]
!108 = !MDLocation(line: 55, scope: !54)
!109 = !{!"0x100\00return_val\0056\000", !54, !7, !12} ; [ DW_TAG_auto_variable ] [return_val] [line 56]
!110 = !MDLocation(line: 56, scope: !54)
!111 = !MDLocation(line: 56, scope: !112)
!112 = !{!"0xb\0056\000\001", !1, !54} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/sret.cpp]
!113 = !{!"0x100\00a\0058\000", !54, !7, !4} ; [ DW_TAG_auto_variable ] [a] [line 58]
!114 = !MDLocation(line: 58, scope: !54)
!115 = !MDLocation(line: 59, scope: !54)
!116 = !MDLocation(line: 60, scope: !54)
!117 = !MDLocation(line: 60, scope: !118)
!118 = !{!"0xb\0060\000\001", !1, !54} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/sret.cpp]
!119 = !MDLocation(line: 60, scope: !120)
!120 = !{!"0xb\0060\000\003", !1, !54} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/sret.cpp]
!121 = !MDLocation(line: 60, scope: !122)
!122 = !{!"0xb\0060\000\002", !1, !54} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/sret.cpp]
!123 = !{!"0x101\00this\0016777216\001088", !62, null, !90} ; [ DW_TAG_arg_variable ] [this] [line 0]
!124 = !MDLocation(line: 0, scope: !62)
!125 = !MDLocation(line: 41, scope: !62)
!126 = !{!"0x101\00this\0016777216\001088", !61, null, !68} ; [ DW_TAG_arg_variable ] [this] [line 0]
!127 = !MDLocation(line: 0, scope: !61)
!128 = !MDLocation(line: 8, scope: !61)
!129 = !MDLocation(line: 8, scope: !130)
!130 = !{!"0xb\008\000\001", !1, !61} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/sret.cpp]
!131 = !MDLocation(line: 8, scope: !132)
!132 = !{!"0xb\008\000\002", !1, !61} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/sret.cpp]
!133 = !MDLocation(line: 8, scope: !134)
!134 = !{!"0xb\008\000\003", !1, !61} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/sret.cpp]
