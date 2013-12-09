; REQUIRES: object-emission

; RUN: llc -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; This was pulled from clang's debug-info-template-recursive.cpp test.
; class base { };

; template <class T> class foo : public base  {
;   void operator=(const foo r) { }
; };

; class bar : public foo<void> { };
; bar filters;

; CHECK: DW_TAG_template_type_parameter [{{.*}}]
; CHECK-NEXT: DW_AT_name{{.*}}"T"
; CHECK-NOT: DW_AT_type
; CHECK: NULL

%class.bar = type { i8 }

@filters = global %class.bar zeroinitializer, align 1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!36, !37}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.4 (trunk 187958) (llvm/trunk 187964)", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !2, metadata !3, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/tmp/debug-info-template-recursive.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"debug-info-template-recursive.cpp", metadata !"/usr/local/google/home/echristo/tmp"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786484, i32 0, null, metadata !"filters", metadata !"filters", metadata !"", metadata !5, i32 10, metadata !6, i32 0, i32 1, %class.bar* @filters, null} ; [ DW_TAG_variable ] [filters] [line 10] [def]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/usr/local/google/home/echristo/tmp/debug-info-template-recursive.cpp]
!6 = metadata !{i32 786434, metadata !1, null, metadata !"bar", i32 9, i64 8, i64 8, i32 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_class_type ] [bar] [line 9, size 8, align 8, offset 0] [def] [from ]
!7 = metadata !{metadata !8, metadata !31}
!8 = metadata !{i32 786460, null, metadata !6, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !9} ; [ DW_TAG_inheritance ] [line 0, size 0, align 0, offset 0] [from foo<void>]
!9 = metadata !{i32 786434, metadata !1, null, metadata !"foo<void>", i32 5, i64 8, i64 8, i32 0, i32 0, null, metadata !10, i32 0, null, metadata !29, null} ; [ DW_TAG_class_type ] [foo<void>] [line 5, size 8, align 8, offset 0] [def] [from ]
!10 = metadata !{metadata !11, metadata !19, metadata !25}
!11 = metadata !{i32 786460, null, metadata !9, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !12} ; [ DW_TAG_inheritance ] [line 0, size 0, align 0, offset 0] [from base]
!12 = metadata !{i32 786434, metadata !1, null, metadata !"base", i32 3, i64 8, i64 8, i32 0, i32 0, null, metadata !13, i32 0, null, null, null} ; [ DW_TAG_class_type ] [base] [line 3, size 8, align 8, offset 0] [def] [from ]
!13 = metadata !{metadata !14}
!14 = metadata !{i32 786478, metadata !1, metadata !12, metadata !"base", metadata !"base", metadata !"", i32 3, metadata !15, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !18, i32 3} ; [ DW_TAG_subprogram ] [line 3] [base]
!15 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !16, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = metadata !{null, metadata !17}
!17 = metadata !{i32 786447, i32 0, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !12} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from base]
!18 = metadata !{i32 786468}
!19 = metadata !{i32 786478, metadata !1, metadata !9, metadata !"operator=", metadata !"operator=", metadata !"_ZN3fooIvEaSES0_", i32 6, metadata !20, i1 false, i1 false, i32 0, i32 0, null, i32 257, i1 false, null, null, i32 0, metadata !24, i32 6} ; [ DW_TAG_subprogram ] [line 6] [private] [operator=]
!20 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !21, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!21 = metadata !{null, metadata !22, metadata !23}
!22 = metadata !{i32 786447, i32 0, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from foo<void>]
!23 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !9} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from foo<void>]
!24 = metadata !{i32 786468}
!25 = metadata !{i32 786478, metadata !1, metadata !9, metadata !"foo", metadata !"foo", metadata !"", i32 5, metadata !26, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !28, i32 5} ; [ DW_TAG_subprogram ] [line 5] [foo]
!26 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !27, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!27 = metadata !{null, metadata !22}
!28 = metadata !{i32 786468}
!29 = metadata !{metadata !30}
!30 = metadata !{i32 786479, null, metadata !"T", null, null, i32 0, i32 0} ; [ DW_TAG_template_type_parameter ]
!31 = metadata !{i32 786478, metadata !1, metadata !6, metadata !"bar", metadata !"bar", metadata !"", i32 9, metadata !32, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !35, i32 9} ; [ DW_TAG_subprogram ] [line 9] [bar]
!32 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !33, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!33 = metadata !{null, metadata !34}
!34 = metadata !{i32 786447, i32 0, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !6} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from bar]
!35 = metadata !{i32 786468}
!36 = metadata !{i32 2, metadata !"Dwarf Version", i32 3}
!37 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
