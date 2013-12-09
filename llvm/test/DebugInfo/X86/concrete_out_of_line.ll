; RUN: llc -mtriple=x86_64-linux %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; test that we add DW_AT_inline even when we only have concrete out of line
; instances.

; first check that we have a TAG_subprogram at a given offset and it has
; AT_inline.

; CHECK: 0x0000011c:   DW_TAG_subprogram [17]
; CHECK-NEXT:     DW_AT_specification
; CHECK-NEXT:     DW_AT_inline


; and then that a TAG_subprogram refers to it with AT_abstract_origin.

; CHECK: 0x0000015d:   DW_TAG_subprogram [19]
; CHECK-NEXT: DW_AT_abstract_origin [DW_FORM_ref4]    (cu + 0x011c => {0x0000011c})

define i32 @_ZN17nsAutoRefCnt7ReleaseEv() {
entry:
  store i32 1, i32* null, align 4, !dbg !50
  tail call void @_Z8moz_freePv(i8* null) nounwind, !dbg !54
  ret i32 0
}

define void @_ZN17nsAutoRefCntD1Ev() {
entry:
  tail call void @_Z8moz_freePv(i8* null) nounwind, !dbg !57
  ret void
}

declare void @_Z8moz_freePv(i8*)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!60}

!0 = metadata !{i32 786449, metadata !59, i32 4, metadata !"clang version 3.1 ()", i1 true, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !47,  metadata !47, metadata !""} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5, metadata !23, metadata !27, metadata !31}
!5 = metadata !{i32 720942, metadata !6, null, metadata !"Release", metadata !"Release", metadata !"_ZN17nsAutoRefCnt7ReleaseEv", i32 14, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32* null, null, metadata !12, metadata !20, i32 14} ; [ DW_TAG_subprogram ] [line 14] [def] [Release]
!6 = metadata !{i32 720937, metadata !59} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 720917, i32 0, null, i32 0, i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9, metadata !10}
!9 = metadata !{i32 720932, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!10 = metadata !{i32 786447, i32 0, null, i32 0, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !11} ; [ DW_TAG_pointer_type ]
!11 = metadata !{i32 786451, metadata !59, null, metadata !"nsAutoRefCnt", i32 10, i64 0, i64 0, i32 0, i32 4, null, null, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [nsAutoRefCnt] [line 10, size 0, align 0, offset 0] [decl] [from ]
!12 = metadata !{i32 720942, metadata !6, metadata !13, metadata !"Release", metadata !"Release", metadata !"_ZN17nsAutoRefCnt7ReleaseEv", i32 11, metadata !7, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 true, null, null, i32 0, metadata !18, i32 11} ; [ DW_TAG_subprogram ]
!13 = metadata !{i32 720898, metadata !59, null, metadata !"nsAutoRefCnt", i32 10, i64 8, i64 8, i32 0, i32 0, null, metadata !14, null, null, null} ; [ DW_TAG_class_type ]
!14 = metadata !{metadata !12, metadata !15}
!15 = metadata !{i32 720942, metadata !6, metadata !13, metadata !"~nsAutoRefCnt", metadata !"~nsAutoRefCnt", metadata !"", i32 12, metadata !16, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 true, null, null, i32 0, metadata !18, i32 12} ; [ DW_TAG_subprogram ]
!16 = metadata !{i32 720917, i32 0, null, i32 0, i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !17, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!17 = metadata !{null, metadata !10}
!18 = metadata !{i32 720932}                      ; [ DW_TAG_base_type ]
!20 = metadata !{metadata !22}
!22 = metadata !{i32 786689, metadata !5, metadata !"this", metadata !6, i32 16777230, metadata !10, i32 64, i32 0} ; [ DW_TAG_arg_variable ]
!23 = metadata !{i32 720942, metadata !6, null, metadata !"~nsAutoRefCnt", metadata !"~nsAutoRefCnt", metadata !"_ZN17nsAutoRefCntD1Ev", i32 18, metadata !16, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32* null, null, metadata !15, metadata !24, i32 18} ; [ DW_TAG_subprogram ] [line 18] [def] [~nsAutoRefCnt]
!24 = metadata !{metadata !26}
!26 = metadata !{i32 786689, metadata !23, metadata !"this", metadata !6, i32 16777234, metadata !10, i32 64, i32 0} ; [ DW_TAG_arg_variable ]
!27 = metadata !{i32 720942, metadata !6, null, metadata !"~nsAutoRefCnt", metadata !"~nsAutoRefCnt", metadata !"_ZN17nsAutoRefCntD2Ev", i32 18, metadata !16, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32* null, null, metadata !15, metadata !28, i32 18} ; [ DW_TAG_subprogram ] [line 18] [def] [~nsAutoRefCnt]
!28 = metadata !{metadata !30}
!30 = metadata !{i32 786689, metadata !27, metadata !"this", metadata !6, i32 16777234, metadata !10, i32 64, i32 0} ; [ DW_TAG_arg_variable ]
!31 = metadata !{i32 720942, metadata !6, null, metadata !"operator=", metadata !"operator=", metadata !"_ZN12nsAutoRefCntaSEi", i32 4, metadata !32, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, null, null, metadata !36, metadata !43, i32 4} ; [ DW_TAG_subprogram ] [line 4] [def] [operator=]
!32 = metadata !{i32 720917, i32 0, null, i32 0, i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !33, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!33 = metadata !{metadata !9, metadata !34, metadata !9}
!34 = metadata !{i32 786447, i32 0, null, i32 0, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !35} ; [ DW_TAG_pointer_type ]
!35 = metadata !{i32 786451, metadata !59, null, metadata !"nsAutoRefCnt", i32 2, i64 0, i64 0, i32 0, i32 4, null, null, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [nsAutoRefCnt] [line 2, size 0, align 0, offset 0] [decl] [from ]
!36 = metadata !{i32 720942, metadata !6, metadata !37, metadata !"operator=", metadata !"operator=", metadata !"_ZN12nsAutoRefCntaSEi", i32 4, metadata !32, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 true, null, null, i32 0, metadata !18, i32 4} ; [ DW_TAG_subprogram ]
!37 = metadata !{i32 720898, metadata !59, null, metadata !"nsAutoRefCnt", i32 2, i64 32, i64 32, i32 0, i32 0, null, metadata !38, i32 0, null, null, null} ; [ DW_TAG_class_type ] [nsAutoRefCnt] [line 2, size 32, align 32, offset 0] [def] [from ]
!38 = metadata !{metadata !39, metadata !40, metadata !36}
!39 = metadata !{i32 786445, metadata !59, metadata !37, metadata !"mValue", i32 7, i64 32, i64 32, i64 0, i32 0, metadata !9} ; [ DW_TAG_member ]
!40 = metadata !{i32 720942, metadata !6, metadata !37, metadata !"nsAutoRefCnt", metadata !"nsAutoRefCnt", metadata !"", i32 3, metadata !41, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 true, null, null, i32 0, metadata !18, i32 3} ; [ DW_TAG_subprogram ]
!41 = metadata !{i32 720917, i32 0, null, i32 0, i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !42, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!42 = metadata !{null, metadata !34}
!43 = metadata !{metadata !45, metadata !46}
!45 = metadata !{i32 786689, metadata !31, metadata !"this", metadata !6, i32 16777220, metadata !34, i32 64, i32 0} ; [ DW_TAG_arg_variable ]
!46 = metadata !{i32 786689, metadata !31, metadata !"aValue", metadata !6, i32 33554436, metadata !9, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!47 = metadata !{metadata !49}
!49 = metadata !{i32 720948, i32 0, null, metadata !"mRefCnt", metadata !"mRefCnt", metadata !"", metadata !6, i32 9, metadata !37, i32 0, i32 1, i32* null, null} ; [ DW_TAG_variable ]
!50 = metadata !{i32 5, i32 5, metadata !51, metadata !52}
!51 = metadata !{i32 786443, metadata !6, metadata !31, i32 4, i32 29, i32 2} ; [ DW_TAG_lexical_block ]
!52 = metadata !{i32 15, i32 0, metadata !53, null}
!53 = metadata !{i32 786443, metadata !6, metadata !5, i32 14, i32 34, i32 0} ; [ DW_TAG_lexical_block ]
!54 = metadata !{i32 19, i32 3, metadata !55, metadata !56}
!55 = metadata !{i32 786443, metadata !6, metadata !27, i32 18, i32 41, i32 1} ; [ DW_TAG_lexical_block ]
!56 = metadata !{i32 18, i32 41, metadata !23, metadata !52}
!57 = metadata !{i32 19, i32 3, metadata !55, metadata !58}
!58 = metadata !{i32 18, i32 41, metadata !23, null}
!59 = metadata !{metadata !"nsAutoRefCnt.ii", metadata !"/Users/espindola/mozilla-central/obj-x86_64-apple-darwin11.2.0/netwerk/base/src"}
!60 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
