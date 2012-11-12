; RUN: llc -mtriple=x86_64-linux %s -o %t -filetype=obj
; RUN: llvm-dwarfdump %t | FileCheck %s

; test that we add DW_AT_inline even when we only have concrete out of line
; instances.

; first check that we have a TAG_subprogram at a given offset and it has
; AT_inline.

; CHECK: 0x0000011e:   DW_TAG_subprogram [18]
; CHECK-NEXT:     DW_AT_specification
; CHECK-NEXT:     DW_AT_inline


; and then that a TAG_subprogram refers to it with AT_abstract_origin.

; CHECK: 0x0000015f:   DW_TAG_subprogram [20]
; CHECK-NEXT: DW_AT_abstract_origin [DW_FORM_ref4]    (cu + 0x011e => {0x0000011e})

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

!0 = metadata !{i32 720913, i32 0, i32 4, metadata !"nsAutoRefCnt.cpp", metadata !"/Users/espindola/mozilla-central/obj-x86_64-apple-darwin11.2.0/netwerk/base/src", metadata !"clang version 3.1 ()", i1 true, i1 true, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !47} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !5, metadata !23, metadata !27, metadata !31}
!5 = metadata !{i32 720942, i32 0, null, metadata !"Release", metadata !"Release", metadata !"_ZN17nsAutoRefCnt7ReleaseEv", metadata !6, i32 14, metadata !7, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 true, i32* null, null, metadata !12, metadata !20} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 720937, metadata !"nsAutoRefCnt.ii", metadata !"/Users/espindola/mozilla-central/obj-x86_64-apple-darwin11.2.0/netwerk/base/src", null} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 720917, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !8, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!8 = metadata !{metadata !9, metadata !10}
!9 = metadata !{i32 720932, null, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!10 = metadata !{i32 720911, i32 0, metadata !"", i32 0, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !11} ; [ DW_TAG_pointer_type ]
!11 = metadata !{i32 720915, null, metadata !"nsAutoRefCnt", metadata !6, i32 10, i64 0, i64 0, i32 0, i32 4, i32 0, null, i32 0, i32 0} ; [ DW_TAG_structure_type ]
!12 = metadata !{i32 720942, i32 0, metadata !13, metadata !"Release", metadata !"Release", metadata !"_ZN17nsAutoRefCnt7ReleaseEv", metadata !6, i32 11, metadata !7, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 true, null, null, i32 0, metadata !18} ; [ DW_TAG_subprogram ]
!13 = metadata !{i32 720898, null, metadata !"nsAutoRefCnt", metadata !6, i32 10, i64 8, i64 8, i32 0, i32 0, null, metadata !14, i32 0, null, null} ; [ DW_TAG_class_type ]
!14 = metadata !{metadata !12, metadata !15}
!15 = metadata !{i32 720942, i32 0, metadata !13, metadata !"~nsAutoRefCnt", metadata !"~nsAutoRefCnt", metadata !"", metadata !6, i32 12, metadata !16, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 true, null, null, i32 0, metadata !18} ; [ DW_TAG_subprogram ]
!16 = metadata !{i32 720917, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !17, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!17 = metadata !{null, metadata !10}
!18 = metadata !{metadata !19}
!19 = metadata !{i32 720932}                      ; [ DW_TAG_base_type ]
!20 = metadata !{metadata !21}
!21 = metadata !{metadata !22}
!22 = metadata !{i32 721153, metadata !5, metadata !"this", metadata !6, i32 16777230, metadata !10, i32 64, i32 0} ; [ DW_TAG_arg_variable ]
!23 = metadata !{i32 720942, i32 0, null, metadata !"~nsAutoRefCnt", metadata !"~nsAutoRefCnt", metadata !"_ZN17nsAutoRefCntD1Ev", metadata !6, i32 18, metadata !16, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 true, i32* null, null, metadata !15, metadata !24} ; [ DW_TAG_subprogram ]
!24 = metadata !{metadata !25}
!25 = metadata !{metadata !26}
!26 = metadata !{i32 721153, metadata !23, metadata !"this", metadata !6, i32 16777234, metadata !10, i32 64, i32 0} ; [ DW_TAG_arg_variable ]
!27 = metadata !{i32 720942, i32 0, null, metadata !"~nsAutoRefCnt", metadata !"~nsAutoRefCnt", metadata !"_ZN17nsAutoRefCntD2Ev", metadata !6, i32 18, metadata !16, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 true, i32* null, null, metadata !15, metadata !28} ; [ DW_TAG_subprogram ]
!28 = metadata !{metadata !29}
!29 = metadata !{metadata !30}
!30 = metadata !{i32 721153, metadata !27, metadata !"this", metadata !6, i32 16777234, metadata !10, i32 64, i32 0} ; [ DW_TAG_arg_variable ]
!31 = metadata !{i32 720942, i32 0, null, metadata !"operator=", metadata !"operator=", metadata !"_ZN12nsAutoRefCntaSEi", metadata !6, i32 4, metadata !32, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 true, null, null, metadata !36, metadata !43} ; [ DW_TAG_subprogram ]
!32 = metadata !{i32 720917, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !33, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!33 = metadata !{metadata !9, metadata !34, metadata !9}
!34 = metadata !{i32 720911, i32 0, metadata !"", i32 0, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !35} ; [ DW_TAG_pointer_type ]
!35 = metadata !{i32 720915, null, metadata !"nsAutoRefCnt", metadata !6, i32 2, i64 0, i64 0, i32 0, i32 4, i32 0, null, i32 0, i32 0} ; [ DW_TAG_structure_type ]
!36 = metadata !{i32 720942, i32 0, metadata !37, metadata !"operator=", metadata !"operator=", metadata !"_ZN12nsAutoRefCntaSEi", metadata !6, i32 4, metadata !32, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 true, null, null, i32 0, metadata !18} ; [ DW_TAG_subprogram ]
!37 = metadata !{i32 720898, null, metadata !"nsAutoRefCnt", metadata !6, i32 2, i64 32, i64 32, i32 0, i32 0, null, metadata !38, i32 0, null, null} ; [ DW_TAG_class_type ]
!38 = metadata !{metadata !39, metadata !40, metadata !36}
!39 = metadata !{i32 720909, metadata !37, metadata !"mValue", metadata !6, i32 7, i64 32, i64 32, i64 0, i32 0, metadata !9} ; [ DW_TAG_member ]
!40 = metadata !{i32 720942, i32 0, metadata !37, metadata !"nsAutoRefCnt", metadata !"nsAutoRefCnt", metadata !"", metadata !6, i32 3, metadata !41, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 true, null, null, i32 0, metadata !18} ; [ DW_TAG_subprogram ]
!41 = metadata !{i32 720917, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !42, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!42 = metadata !{null, metadata !34}
!43 = metadata !{metadata !44}
!44 = metadata !{metadata !45, metadata !46}
!45 = metadata !{i32 721153, metadata !31, metadata !"this", metadata !6, i32 16777220, metadata !34, i32 64, i32 0} ; [ DW_TAG_arg_variable ]
!46 = metadata !{i32 721153, metadata !31, metadata !"aValue", metadata !6, i32 33554436, metadata !9, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!47 = metadata !{metadata !48}
!48 = metadata !{metadata !49}
!49 = metadata !{i32 720948, i32 0, null, metadata !"mRefCnt", metadata !"mRefCnt", metadata !"", metadata !6, i32 9, metadata !37, i32 0, i32 1, i32* null} ; [ DW_TAG_variable ]
!50 = metadata !{i32 5, i32 5, metadata !51, metadata !52}
!51 = metadata !{i32 720907, metadata !31, i32 4, i32 29, metadata !6, i32 2} ; [ DW_TAG_lexical_block ]
!52 = metadata !{i32 15, i32 0, metadata !53, null}
!53 = metadata !{i32 720907, metadata !5, i32 14, i32 34, metadata !6, i32 0} ; [ DW_TAG_lexical_block ]
!54 = metadata !{i32 19, i32 3, metadata !55, metadata !56}
!55 = metadata !{i32 720907, metadata !27, i32 18, i32 41, metadata !6, i32 1} ; [ DW_TAG_lexical_block ]
!56 = metadata !{i32 18, i32 41, metadata !23, metadata !52}
!57 = metadata !{i32 19, i32 3, metadata !55, metadata !58}
!58 = metadata !{i32 18, i32 41, metadata !23, null}
