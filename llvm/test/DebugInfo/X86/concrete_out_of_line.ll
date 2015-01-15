; RUN: llc -mtriple=x86_64-linux < %s -filetype=obj | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; test that we add DW_AT_inline even when we only have concrete out of line
; instances.

; first check that we have a TAG_subprogram at a given offset and it has
; AT_inline.

; CHECK: DW_TAG_class_type
; CHECK:   DW_TAG_subprogram
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_MIPS_linkage_name {{.*}} "_ZN12nsAutoRefCntaSEi"

; CHECK: DW_TAG_class_type
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_MIPS_linkage_name {{.*}} "_ZN17nsAutoRefCnt7ReleaseEv"
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}} "~nsAutoRefCnt"

; CHECK: DW_TAG_subprogram
; CHECK-NEXT:     DW_AT_{{.*}}linkage_name {{.*}}D2
; CHECK-NEXT:     DW_AT_specification {{.*}} "~nsAutoRefCnt"
; CHECK-NEXT:     DW_AT_inline
; CHECK-NOT:      DW_AT
; CHECK: DW_TAG
; CHECK: DW_TAG_subprogram
; CHECK-NEXT:     DW_AT_{{.*}}linkage_name {{.*}}D1
; CHECK-NEXT:     DW_AT_specification {{.*}} "~nsAutoRefCnt"
; CHECK-NEXT:     DW_AT_inline
; CHECK-NOT:     DW_AT
; CHECK: [[D1_THIS_ABS:.*]]: DW_TAG_formal_parameter

; CHECK: DW_TAG_subprogram
; CHECK:     DW_AT_specification {{.*}} "_ZN17nsAutoRefCnt7ReleaseEv"
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: NULL
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_inlined_subroutine
; CHECK-NEXT: DW_AT_abstract_origin {{.*}} "_ZN12nsAutoRefCntaSEi"
; CHECK-NOT: NULL
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_inlined_subroutine
; CHECK-NEXT: DW_AT_abstract_origin {{.*}} "_ZN17nsAutoRefCntD1Ev"
; CHECK-NOT: NULL
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_inlined_subroutine
; CHECK-NEXT: DW_AT_abstract_origin {{.*}} "_ZN17nsAutoRefCntD2Ev"

; and then that a TAG_subprogram refers to it with AT_abstract_origin.

; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_abstract_origin {{.*}} "_ZN17nsAutoRefCntD1Ev"
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_abstract_origin {{.*}} {[[D1_THIS_ABS]]} "this"
; CHECK: DW_TAG_inlined_subroutine
; CHECK-NEXT: DW_AT_abstract_origin {{.*}} "_ZN17nsAutoRefCntD2Ev"


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

!0 = !{!"0x11\004\00clang version 3.1 ()\001\00\000\00\000", !59, !1, !1, !3, !47,  !1} ; [ DW_TAG_compile_unit ]
!1 = !{}
!3 = !{!5, !23, !27, !31}
!5 = !{!"0x2e\00Release\00Release\00_ZN17nsAutoRefCnt7ReleaseEv\0014\000\001\000\006\00256\001\0014", !6, null, !7, null, i32 ()* @_ZN17nsAutoRefCnt7ReleaseEv , null, !12, !20} ; [ DW_TAG_subprogram ] [line 14] [def] [Release]
!6 = !{!"0x29", !59} ; [ DW_TAG_file_type ]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{!9, !10}
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!10 = !{!"0xf\00\000\0064\0064\000\0064", i32 0, null, !11} ; [ DW_TAG_pointer_type ]
!11 = !{!"0x13\00nsAutoRefCnt\0010\000\000\000\004\000", !59, null, null, null, null, null, null} ; [ DW_TAG_structure_type ] [nsAutoRefCnt] [line 10, size 0, align 0, offset 0] [decl] [from ]
!12 = !{!"0x2e\00Release\00Release\00_ZN17nsAutoRefCnt7ReleaseEv\0011\000\000\000\006\00256\001\0011", !6, !13, !7, null, null, null, i32 0, !18} ; [ DW_TAG_subprogram ]
!13 = !{!"0x2\00nsAutoRefCnt\0010\008\008\000\000\000", !59, null, null, !14, null, null} ; [ DW_TAG_class_type ]
!14 = !{!12, !15}
!15 = !{!"0x2e\00~nsAutoRefCnt\00~nsAutoRefCnt\00\0012\000\000\000\006\00256\001\0012", !6, !13, !16, null, null, null, i32 0, !18} ; [ DW_TAG_subprogram ]
!16 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !17, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!17 = !{null, !10}
!18 = !{}
!20 = !{!22}
!22 = !{!"0x101\00this\0016777230\0064", !5, !6, !10} ; [ DW_TAG_arg_variable ]
!23 = !{!"0x2e\00~nsAutoRefCnt\00~nsAutoRefCnt\00_ZN17nsAutoRefCntD1Ev\0018\000\001\000\006\00256\001\0018", !6, null, !16, null, void ()* @_ZN17nsAutoRefCntD1Ev, null, !15, !24} ; [ DW_TAG_subprogram ] [line 18] [def] [~nsAutoRefCnt]
!24 = !{!26}
!26 = !{!"0x101\00this\0016777234\0064", !23, !6, !10} ; [ DW_TAG_arg_variable ]
!27 = !{!"0x2e\00~nsAutoRefCnt\00~nsAutoRefCnt\00_ZN17nsAutoRefCntD2Ev\0018\000\001\000\006\00256\001\0018", !6, null, !16, null, i32* null, null, !15, !28} ; [ DW_TAG_subprogram ] [line 18] [def] [~nsAutoRefCnt]
!28 = !{!30}
!30 = !{!"0x101\00this\0016777234\0064", !27, !6, !10} ; [ DW_TAG_arg_variable ]
!31 = !{!"0x2e\00operator=\00operator=\00_ZN12nsAutoRefCntaSEi\004\000\001\000\006\00256\001\004", !6, null, !32, null, null, null, !36, !43} ; [ DW_TAG_subprogram ] [line 4] [def] [operator=]
!32 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !33, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!33 = !{!9, !34, !9}
!34 = !{!"0xf\00\000\0064\0064\000\0064", i32 0, null, !35} ; [ DW_TAG_pointer_type ]
!35 = !{!"0x13\00nsAutoRefCnt\002\000\000\000\004\000", !59, null, null, null, null, null, null} ; [ DW_TAG_structure_type ] [nsAutoRefCnt] [line 2, size 0, align 0, offset 0] [decl] [from ]
!36 = !{!"0x2e\00operator=\00operator=\00_ZN12nsAutoRefCntaSEi\004\000\000\000\006\00256\001\004", !6, !37, !32, null, null, null, i32 0, !18} ; [ DW_TAG_subprogram ]
!37 = !{!"0x2\00nsAutoRefCnt\002\0032\0032\000\000\000", !59, null, null, !38, null, null, null} ; [ DW_TAG_class_type ] [nsAutoRefCnt] [line 2, size 32, align 32, offset 0] [def] [from ]
!38 = !{!39, !40, !36}
!39 = !{!"0xd\00mValue\007\0032\0032\000\000", !59, !37, !9} ; [ DW_TAG_member ]
!40 = !{!"0x2e\00nsAutoRefCnt\00nsAutoRefCnt\00\003\000\000\000\006\00256\001\003", !6, !37, !41, null, null, null, i32 0, !18} ; [ DW_TAG_subprogram ]
!41 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !42, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!42 = !{null, !34}
!43 = !{!45, !46}
!45 = !{!"0x101\00this\0016777220\0064", !31, !6, !34} ; [ DW_TAG_arg_variable ]
!46 = !{!"0x101\00aValue\0033554436\000", !31, !6, !9} ; [ DW_TAG_arg_variable ]
!47 = !{!49}
!49 = !{!"0x34\00mRefCnt\00mRefCnt\00\009\000\001", null, !6, !37, i32* null, null} ; [ DW_TAG_variable ]
!50 = !MDLocation(line: 5, column: 5, scope: !51, inlinedAt: !52)
!51 = !{!"0xb\004\0029\002", !6, !31} ; [ DW_TAG_lexical_block ]
!52 = !MDLocation(line: 15, scope: !53)
!53 = !{!"0xb\0014\0034\000", !6, !5} ; [ DW_TAG_lexical_block ]
!54 = !MDLocation(line: 19, column: 3, scope: !55, inlinedAt: !56)
!55 = !{!"0xb\0018\0041\001", !6, !27} ; [ DW_TAG_lexical_block ]
!56 = !MDLocation(line: 18, column: 41, scope: !23, inlinedAt: !52)
!57 = !MDLocation(line: 19, column: 3, scope: !55, inlinedAt: !58)
!58 = !MDLocation(line: 18, column: 41, scope: !23)
!59 = !{!"nsAutoRefCnt.ii", !"/Users/espindola/mozilla-central/obj-x86_64-apple-darwin11.2.0/netwerk/base/src"}
!60 = !{i32 1, !"Debug Info Version", i32 2}
