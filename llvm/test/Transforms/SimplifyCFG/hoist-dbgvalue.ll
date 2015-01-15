; RUN: opt -simplifycfg -S < %s | FileCheck %s

define i32 @foo(i32 %i) nounwind ssp {
  call void @llvm.dbg.value(metadata i32 %i, i64 0, metadata !6, metadata !{}), !dbg !7
  call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !9, metadata !{}), !dbg !11
  %1 = icmp ne i32 %i, 0, !dbg !12
;CHECK: call i32 (...)* @bar()
;CHECK-NEXT: llvm.dbg.value
  br i1 %1, label %2, label %4, !dbg !12

; <label>:2                                       ; preds = %0
  %3 = call i32 (...)* @bar(), !dbg !13
  call void @llvm.dbg.value(metadata i32 %3, i64 0, metadata !9, metadata !{}), !dbg !13
  br label %6, !dbg !15

; <label>:4                                       ; preds = %0
  %5 = call i32 (...)* @bar(), !dbg !16
  call void @llvm.dbg.value(metadata i32 %5, i64 0, metadata !9, metadata !{}), !dbg !16
  br label %6, !dbg !18

; <label>:6                                       ; preds = %4, %2
  %k.0 = phi i32 [ %3, %2 ], [ %5, %4 ]
  ret i32 %k.0, !dbg !19
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare i32 @bar(...)

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.module.flags = !{!21}
!llvm.dbg.sp = !{!0}

!0 = !{!"0x2e\00foo\00foo\00\002\000\001\000\006\00256\000\000", !20, !1, !3, null, i32 (i32)* @foo, null, null, null} ; [ DW_TAG_subprogram ] [line 2] [def] [scope 0] [foo]
!1 = !{!"0x29", !20} ; [ DW_TAG_file_type ]
!2 = !{!"0x11\0012\00clang\001\00\000\00\000", !20, !8, !8, null, null, null} ; [ DW_TAG_compile_unit ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !20, !1, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{!5}
!5 = !{!"0x24\00int\000\0032\0032\000\000\005", null, !2} ; [ DW_TAG_base_type ]
!6 = !{!"0x101\00i\0016777218\000", !0, !1, !5} ; [ DW_TAG_arg_variable ]
!7 = !MDLocation(line: 2, column: 13, scope: !0)
!8 = !{i32 0}
!9 = !{!"0x100\00k\003\000", !10, !1, !5} ; [ DW_TAG_auto_variable ]
!10 = !{!"0xb\002\0016\000", !20, !0} ; [ DW_TAG_lexical_block ]
!11 = !MDLocation(line: 3, column: 12, scope: !10)
!12 = !MDLocation(line: 4, column: 3, scope: !10)
!13 = !MDLocation(line: 5, column: 5, scope: !14)
!14 = !{!"0xb\004\0010\001", !20, !10} ; [ DW_TAG_lexical_block ]
!15 = !MDLocation(line: 6, column: 3, scope: !14)
!16 = !MDLocation(line: 7, column: 5, scope: !17)
!17 = !{!"0xb\006\0010\002", !20, !10} ; [ DW_TAG_lexical_block ]
!18 = !MDLocation(line: 8, column: 3, scope: !17)
!19 = !MDLocation(line: 9, column: 3, scope: !10)
!20 = !{!"b.c", !"/private/tmp"}
!21 = !{i32 1, !"Debug Info Version", i32 2}
