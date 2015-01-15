; RUN: llc %s -filetype=obj -o %t.o
; RUN: llvm-dwarfdump -debug-dump=info %t.o | FileCheck %s
; Test emitting a constant for an aggregate type.
;
; clang -S -O1 -emit-llvm
;
; typedef struct { unsigned i; } S;
;
; unsigned foo(S s) {
;   s.i = 1;
;   return s.i;
; }
;
; class C { public: unsigned i; };
;
; unsigned foo(C c) {
;   c.i = 2;
;   return c.i;
; }
;
; unsigned bar() {
;  int a[1] = { 3 };
;   return a[0];
; }
;
; CHECK:  DW_TAG_formal_parameter
; CHECK-NEXT: DW_AT_const_value [DW_FORM_udata]	(1)
; CHECK-NEXT: DW_AT_name {{.*}} "s"
;
; CHECK:  DW_TAG_formal_parameter
; CHECK-NEXT: DW_AT_const_value [DW_FORM_udata]	(2)
; CHECK-NEXT: DW_AT_name {{.*}} "c"
;
; CHECK:  DW_TAG_variable
; CHECK-NEXT: DW_AT_const_value [DW_FORM_udata]	(3)
; CHECK-NEXT: DW_AT_name {{.*}} "a"

; ModuleID = 'sroasplit-4.cpp'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

; Function Attrs: nounwind readnone ssp uwtable
define i32 @_Z3foo1S(i32 %s.coerce) #0 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %s.coerce, i64 0, metadata !18, metadata !37), !dbg !38
  tail call void @llvm.dbg.value(metadata i32 1, i64 0, metadata !18, metadata !37), !dbg !38
  ret i32 1, !dbg !39
}

; Function Attrs: nounwind readnone ssp uwtable
define i32 @_Z3foo1C(i32 %c.coerce) #0 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %c.coerce, i64 0, metadata !23, metadata !37), !dbg !40
  tail call void @llvm.dbg.value(metadata i32 2, i64 0, metadata !23, metadata !37), !dbg !40
  ret i32 2, !dbg !41
}

; Function Attrs: nounwind readnone ssp uwtable
define i32 @_Z3barv() #0 {
entry:
  tail call void @llvm.dbg.value(metadata i32 3, i64 0, metadata !28, metadata !37), !dbg !42
  ret i32 3, !dbg !43
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind readnone ssp uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!33, !34, !35}
!llvm.ident = !{!36}

!0 = !{!"0x11\004\00clang version 3.6.0 (trunk 225364) (llvm/trunk 225366)\001\00\000\00\001", !1, !2, !3, !11, !2, !2} ; [ DW_TAG_compile_unit ] [/sroasplit-4.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"sroasplit-4.cpp", !""}
!2 = !{}
!3 = !{!4, !8}
!4 = !{!"0x13\00\001\0032\0032\000\000\000", !1, null, null, !5, null, null, !"_ZTS1S"} ; [ DW_TAG_structure_type ] [line 1, size 32, align 32, offset 0] [def] [from ]
!5 = !{!6}
!6 = !{!"0xd\00i\001\0032\0032\000\000", !1, !"_ZTS1S", !7} ; [ DW_TAG_member ] [i] [line 1, size 32, align 32, offset 0] [from unsigned int]
!7 = !{!"0x24\00unsigned int\000\0032\0032\000\000\007", null, null} ; [ DW_TAG_base_type ] [unsigned int] [line 0, size 32, align 32, offset 0, enc DW_ATE_unsigned]
!8 = !{!"0x2\00C\008\0032\0032\000\000\000", !1, null, null, !9, null, null, !"_ZTS1C"} ; [ DW_TAG_class_type ] [C] [line 8, size 32, align 32, offset 0] [def] [from ]
!9 = !{!10}
!10 = !{!"0xd\00i\008\0032\0032\000\003", !1, !"_ZTS1C", !7} ; [ DW_TAG_member ] [i] [line 8, size 32, align 32, offset 0] [public] [from unsigned int]
!11 = !{!12, !19, !24}
!12 = !{!"0x2e\00foo\00foo\00_Z3foo1S\003\000\001\000\000\00256\001\003", !1, !13, !14, null, i32 (i32)* @_Z3foo1S, null, null, !17} ; [ DW_TAG_subprogram ] [line 3] [def] [foo]
!13 = !{!"0x29", !1}                              ; [ DW_TAG_file_type ] [/sroasplit-4.cpp]
!14 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !15, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!15 = !{!7, !16}
!16 = !{!"0x16\00S\001\000\000\000\000", !1, null, !"_ZTS1S"} ; [ DW_TAG_typedef ] [S] [line 1, size 0, align 0, offset 0] [from _ZTS1S]
!17 = !{!18}
!18 = !{!"0x101\00s\0016777219\000", !12, !13, !16} ; [ DW_TAG_arg_variable ] [s] [line 3]
!19 = !{!"0x2e\00foo\00foo\00_Z3foo1C\0010\000\001\000\000\00256\001\0010", !1, !13, !20, null, i32 (i32)* @_Z3foo1C, null, null, !22} ; [ DW_TAG_subprogram ] [line 10] [def] [foo]
!20 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !21, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!21 = !{!7, !"_ZTS1C"}
!22 = !{!23}
!23 = !{!"0x101\00c\0016777226\000", !19, !13, !"_ZTS1C"} ; [ DW_TAG_arg_variable ] [c] [line 10]
!24 = !{!"0x2e\00bar\00bar\00_Z3barv\0015\000\001\000\000\00256\001\0015", !1, !13, !25, null, i32 ()* @_Z3barv, null, null, !27} ; [ DW_TAG_subprogram ] [line 15] [def] [bar]
!25 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !26, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!26 = !{!7}
!27 = !{!28}
!28 = !{!"0x100\00a\0016\000", !24, !13, !29}     ; [ DW_TAG_auto_variable ] [a] [line 16]
!29 = !{!"0x1\00\000\0032\0032\000\000\000", null, null, !30, !31, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 32, align 32, offset 0] [from int]
!30 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!31 = !{!32}
!32 = !{!"0x21\000\001"}                          ; [ DW_TAG_subrange_type ] [0, 0]
!33 = !{i32 2, !"Dwarf Version", i32 2}
!34 = !{i32 2, !"Debug Info Version", i32 2}
!35 = !{i32 1, !"PIC Level", i32 2}
!36 = !{!"clang version 3.6.0 (trunk 225364) (llvm/trunk 225366)"}
!37 = !{!"0x102"}                                 ; [ DW_TAG_expression ]
!38 = !MDLocation(line: 3, column: 16, scope: !12)
!39 = !MDLocation(line: 5, column: 3, scope: !12)
!40 = !MDLocation(line: 10, column: 16, scope: !19)
!41 = !MDLocation(line: 12, column: 3, scope: !19)
!42 = !MDLocation(line: 16, column: 6, scope: !24)
!43 = !MDLocation(line: 17, column: 3, scope: !24)
