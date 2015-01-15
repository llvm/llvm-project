; RUN: opt -simplifycfg -S < %s | FileCheck %s

%0 = type { i32*, i32* }

@0 = external hidden constant [5 x %0], align 4

define void @foo(i32) nounwind ssp {
Entry:
  %1 = icmp slt i32 %0, 0, !dbg !5
  br i1 %1, label %BB5, label %BB1, !dbg !5

BB1:                                              ; preds = %Entry
  %2 = icmp sgt i32 %0, 4, !dbg !5
  br i1 %2, label %BB5, label %BB2, !dbg !5

BB2:                                              ; preds = %BB1
  %3 = shl i32 1, %0, !dbg !5
  %4 = and i32 %3, 31, !dbg !5
  %5 = icmp eq i32 %4, 0, !dbg !5
  br i1 %5, label %BB5, label %BB3, !dbg !5

;CHECK: icmp eq
;CHECK-NEXT: getelementptr
;CHECK-NEXT: icmp eq

BB3:                                              ; preds = %BB2
  %6 = getelementptr inbounds [5 x %0]* @0, i32 0, i32 %0, !dbg !6
  call void @llvm.dbg.value(metadata %0* %6, i64 0, metadata !7, metadata !{}), !dbg !12
  %7 = icmp eq %0* %6, null, !dbg !13
  br i1 %7, label %BB5, label %BB4, !dbg !13

BB4:                                              ; preds = %BB3
  %8 = icmp slt i32 %0, 0, !dbg !5
  ret void, !dbg !14

BB5:                                              ; preds = %BB3, %BB2, %BB1, %Entry
  ret void, !dbg !14
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.sp = !{!0}

!0 = !{!"0x2e\00foo\00foo\00\00231\000\001\000\006\00256\000\000", !15, !1, !3, null, void (i32)* @foo, null, null, null} ; [ DW_TAG_subprogram ] [line 231] [def] [scope 0] [foo]
!1 = !{!"0x29", !15} ; [ DW_TAG_file_type ]
!2 = !{!"0x11\0012\00clang (trunk 129006)\001\00\000\00\000", !15, !4, !4, null, null, null} ; [ DW_TAG_compile_unit ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !15, !1, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{null}
!5 = !MDLocation(line: 131, column: 2, scope: !0)
!6 = !MDLocation(line: 134, column: 2, scope: !0)
!7 = !{!"0x100\00bar\00232\000", !8, !1, !9} ; [ DW_TAG_auto_variable ]
!8 = !{!"0xb\00231\001\003", !15, !0} ; [ DW_TAG_lexical_block ]
!9 = !{!"0xf\00\000\0032\0032\000\000", null, !2, !10} ; [ DW_TAG_pointer_type ]
!10 = !{!"0x26\00\000\000\000\000\000", null, !2, !11} ; [ DW_TAG_const_type ]
!11 = !{!"0x24\00unsigned int\000\0032\0032\000\000\007", null, !2} ; [ DW_TAG_base_type ]
!12 = !MDLocation(line: 232, column: 40, scope: !8)
!13 = !MDLocation(line: 234, column: 2, scope: !8)
!14 = !MDLocation(line: 274, column: 1, scope: !8)
!15 = !{!"a.c", !"/private/tmp"}
