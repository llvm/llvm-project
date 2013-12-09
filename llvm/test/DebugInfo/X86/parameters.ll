; REQUIRES: object-emission
;
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; Test case derived from compiling the following source with clang -g:
;
; namespace pr14763 {
; struct foo {
;   foo(const foo&);
; };
;
; foo func(foo f) {
;   return f; // reference 'f' for now because otherwise we hit another bug
; }
;
; void sink(void*);
;
; void func2(bool b, foo g) {
;   if (b)
;     sink(&g); // reference 'f' for now because otherwise we hit another bug
; }
; }

; CHECK: debug_info contents
; CHECK: DW_AT_name{{.*}} = "f"
; 0x74 is DW_OP_breg4, showing that the parameter is accessed indirectly
; (with a zero offset) from the register parameter
; CHECK: DW_AT_location{{.*}}(<0x0{{.}}> 74 00

; CHECK: DW_AT_name{{.*}} = "g"
; CHECK: DW_AT_location{{.*}}([[G_LOC:0x[0-9]*]])
; CHECK: debug_loc contents
; CHECK-NEXT: [[G_LOC]]: Beginning
; CHECK-NEXT:               Ending
; CHECK-NEXT: Location description: 74 00

%"struct.pr14763::foo" = type { i8 }

; Function Attrs: uwtable
define void @_ZN7pr147634funcENS_3fooE(%"struct.pr14763::foo"* noalias sret %agg.result, %"struct.pr14763::foo"* %f) #0 {
entry:
  call void @llvm.dbg.declare(metadata !{%"struct.pr14763::foo"* %f}, metadata !22), !dbg !24
  call void @_ZN7pr147633fooC1ERKS0_(%"struct.pr14763::foo"* %agg.result, %"struct.pr14763::foo"* %f), !dbg !25
  ret void, !dbg !25
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

declare void @_ZN7pr147633fooC1ERKS0_(%"struct.pr14763::foo"*, %"struct.pr14763::foo"*) #2

; Function Attrs: uwtable
define void @_ZN7pr147635func2EbNS_3fooE(i1 zeroext %b, %"struct.pr14763::foo"* %g) #0 {
entry:
  %b.addr = alloca i8, align 1
  %frombool = zext i1 %b to i8
  store i8 %frombool, i8* %b.addr, align 1
  call void @llvm.dbg.declare(metadata !{i8* %b.addr}, metadata !26), !dbg !27
  call void @llvm.dbg.declare(metadata !{%"struct.pr14763::foo"* %g}, metadata !28), !dbg !27
  %0 = load i8* %b.addr, align 1, !dbg !29
  %tobool = trunc i8 %0 to i1, !dbg !29
  br i1 %tobool, label %if.then, label %if.end, !dbg !29

if.then:                                          ; preds = %entry
  %1 = bitcast %"struct.pr14763::foo"* %g to i8*, !dbg !31
  call void @_ZN7pr147634sinkEPv(i8* %1), !dbg !31
  br label %if.end, !dbg !31

if.end:                                           ; preds = %if.then, %entry
  ret void, !dbg !32
}

declare void @_ZN7pr147634sinkEPv(i8*) #2

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21, !33}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.4 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/tmp/pass.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"pass.cpp", metadata !"/tmp"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4, metadata !17}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"func", metadata !"func", metadata !"_ZN7pr147634funcENS_3fooE", i32 6, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%"struct.pr14763::foo"*, %"struct.pr14763::foo"*)* @_ZN7pr147634funcENS_3fooE, null, null, metadata !2, i32 6} ; [ DW_TAG_subprogram ] [line 6] [def] [func]
!5 = metadata !{i32 786489, metadata !1, null, metadata !"pr14763", i32 1} ; [ DW_TAG_namespace ] [pr14763] [line 1]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8, metadata !8}
!8 = metadata !{i32 786451, metadata !1, metadata !5, metadata !"foo", i32 2, i64 8, i64 8, i32 0, i32 0, null, metadata !9, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [foo] [line 2, size 8, align 8, offset 0] [def] [from ]
!9 = metadata !{metadata !10}
!10 = metadata !{i32 786478, metadata !1, metadata !8, metadata !"foo", metadata !"foo", metadata !"", i32 3, metadata !11, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !16, i32 3} ; [ DW_TAG_subprogram ] [line 3] [foo]
!11 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !12, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = metadata !{null, metadata !13, metadata !14}
!13 = metadata !{i32 786447, i32 0, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !8} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from foo]
!14 = metadata !{i32 786448, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !15} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from ]
!15 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !8} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from foo]
!16 = metadata !{i32 786468}
!17 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"func2", metadata !"func2", metadata !"_ZN7pr147635func2EbNS_3fooE", i32 12, metadata !18, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i1, %"struct.pr14763::foo"*)* @_ZN7pr147635func2EbNS_3fooE, null, null, metadata !2, i32 12} ; [ DW_TAG_subprogram ] [line 12] [def] [func2]
!18 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !19, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!19 = metadata !{null, metadata !20, metadata !8}
!20 = metadata !{i32 786468, null, null, metadata !"bool", i32 0, i64 8, i64 8, i64 0, i32 0, i32 2} ; [ DW_TAG_base_type ] [bool] [line 0, size 8, align 8, offset 0, enc DW_ATE_boolean]
!21 = metadata !{i32 2, metadata !"Dwarf Version", i32 3}
!22 = metadata !{i32 786689, metadata !4, metadata !"f", metadata !23, i32 16777222, metadata !8, i32 8192, i32 0} ; [ DW_TAG_arg_variable ] [f] [line 6]
!23 = metadata !{i32 786473, metadata !1}         ; [ DW_TAG_file_type ] [/tmp/pass.cpp]
!24 = metadata !{i32 6, i32 0, metadata !4, null}
!25 = metadata !{i32 7, i32 0, metadata !4, null}
!26 = metadata !{i32 786689, metadata !17, metadata !"b", metadata !23, i32 16777228, metadata !20, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [b] [line 12]
!27 = metadata !{i32 12, i32 0, metadata !17, null}
!28 = metadata !{i32 786689, metadata !17, metadata !"g", metadata !23, i32 33554444, metadata !8, i32 8192, i32 0} ; [ DW_TAG_arg_variable ] [g] [line 12]
!29 = metadata !{i32 13, i32 0, metadata !30, null}
!30 = metadata !{i32 786443, metadata !1, metadata !17, i32 13, i32 0, i32 0} ; [ DW_TAG_lexical_block ] [/tmp/pass.cpp]
!31 = metadata !{i32 14, i32 0, metadata !30, null}
!32 = metadata !{i32 15, i32 0, metadata !17, null}
!33 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
