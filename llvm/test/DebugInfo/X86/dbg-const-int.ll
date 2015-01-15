; RUN: llc -mtriple=x86_64-apple-darwin12 -filetype=obj < %s \
; RUN:    | llvm-dwarfdump -debug-dump=info - | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.6.7"
; Radar 9511391

; CHECK: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_const_value [DW_FORM_sdata]   (42)
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "i"

define i32 @foo() nounwind uwtable readnone optsize ssp {
entry:
  tail call void @llvm.dbg.value(metadata i32 42, i64 0, metadata !6, metadata !{!"0x102"}), !dbg !9
  ret i32 42, !dbg !10
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!15}

!0 = !{!"0x11\0012\00clang version 3.0 (trunk 132191)\001\00\000\00\000", !13, !14, !14, !11, null,  null} ; [ DW_TAG_compile_unit ]
!1 = !{!"0x2e\00foo\00foo\00\001\000\001\000\006\000\001\000", !13, !2, !3, null, i32 ()* @foo, null, null, !12} ; [ DW_TAG_subprogram ] [line 1] [def] [scope 0] [foo]
!2 = !{!"0x29", !13} ; [ DW_TAG_file_type ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !13, !2, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{!5}
!5 = !{!"0x24\00int\000\0032\0032\000\000\005", null, !0} ; [ DW_TAG_base_type ]
!6 = !{!"0x100\00i\002\000", !7, !2, !5} ; [ DW_TAG_auto_variable ]
!7 = !{!"0xb\001\0011\000", !13, !1} ; [ DW_TAG_lexical_block ]
!8 = !{i32 42}
!9 = !MDLocation(line: 2, column: 12, scope: !7)
!10 = !MDLocation(line: 3, column: 2, scope: !7)
!11 = !{!1}
!12 = !{!6}
!13 = !{!"a.c", !"/private/tmp"}
!14 = !{i32 0}
!15 = !{i32 1, !"Debug Info Version", i32 2}
