; RUN: opt -simplifycfg -S %s | FileCheck %s

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
  call void @llvm.dbg.value(metadata !{%0* %6}, i64 0, metadata !7), !dbg !12
  %7 = icmp eq %0* %6, null, !dbg !13
  br i1 %7, label %BB5, label %BB4, !dbg !13

BB4:                                              ; preds = %BB3
  %8 = icmp slt i32 %0, 0, !dbg !5
  ret void, !dbg !14

BB5:                                              ; preds = %BB3, %BB2, %BB1, %Entry
  ret void, !dbg !14
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.sp = !{!0}

!0 = metadata !{i32 589870, i32 0, metadata !1, metadata !"foo", metadata !"foo", metadata !"", metadata !1, i32 231, metadata !3, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 false, void (i32)* @foo, null} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 589865, metadata !"a.c", metadata !"/private/tmp", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 589841, i32 0, i32 12, metadata !"a.i", metadata !"/private/tmp", metadata !"clang (trunk 129006)", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 589845, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !4, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{null}
!5 = metadata !{i32 131, i32 2, metadata !0, null}
!6 = metadata !{i32 134, i32 2, metadata !0, null}
!7 = metadata !{i32 590080, metadata !8, metadata !"bar", metadata !1, i32 232, metadata !9, i32 0} ; [ DW_TAG_auto_variable ]
!8 = metadata !{i32 589835, metadata !0, i32 231, i32 1, metadata !1, i32 3} ; [ DW_TAG_lexical_block ]
!9 = metadata !{i32 589839, metadata !2, metadata !"", null, i32 0, i64 32, i64 32, i64 0, i32 0, metadata !10} ; [ DW_TAG_pointer_type ]
!10 = metadata !{i32 589862, metadata !2, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !11} ; [ DW_TAG_const_type ]
!11 = metadata !{i32 589860, metadata !2, metadata !"unsigned int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!12 = metadata !{i32 232, i32 40, metadata !8, null}
!13 = metadata !{i32 234, i32 2, metadata !8, null}
!14 = metadata !{i32 274, i32 1, metadata !8, null}
