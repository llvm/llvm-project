; RUN: opt -S -loop-rotate < %s | FileCheck %s

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

define i32 @tak(i32 %x, i32 %y, i32 %z) nounwind ssp {
; CHECK-LABEL: define i32 @tak(
; CHECK: entry
; CHECK-NEXT: call void @llvm.dbg.value(metadata !{i32 %x}

entry:
  br label %tailrecurse

tailrecurse:                                      ; preds = %if.then, %entry
  %x.tr = phi i32 [ %x, %entry ], [ %call, %if.then ]
  %y.tr = phi i32 [ %y, %entry ], [ %call9, %if.then ]
  %z.tr = phi i32 [ %z, %entry ], [ %call14, %if.then ]
  tail call void @llvm.dbg.value(metadata !{i32 %x.tr}, i64 0, metadata !6), !dbg !7
  tail call void @llvm.dbg.value(metadata !{i32 %y.tr}, i64 0, metadata !8), !dbg !9
  tail call void @llvm.dbg.value(metadata !{i32 %z.tr}, i64 0, metadata !10), !dbg !11
  %cmp = icmp slt i32 %y.tr, %x.tr, !dbg !12
  br i1 %cmp, label %if.then, label %if.end, !dbg !12

if.then:                                          ; preds = %tailrecurse
  %sub = sub nsw i32 %x.tr, 1, !dbg !14
  %call = tail call i32 @tak(i32 %sub, i32 %y.tr, i32 %z.tr), !dbg !14
  %sub6 = sub nsw i32 %y.tr, 1, !dbg !14
  %call9 = tail call i32 @tak(i32 %sub6, i32 %z.tr, i32 %x.tr), !dbg !14
  %sub11 = sub nsw i32 %z.tr, 1, !dbg !14
  %call14 = tail call i32 @tak(i32 %sub11, i32 %x.tr, i32 %y.tr), !dbg !14
  br label %tailrecurse

if.end:                                           ; preds = %tailrecurse
  br label %return, !dbg !16

return:                                           ; preds = %if.end
  ret i32 %z.tr, !dbg !17
}

@channelColumns = external global i64
@horzPlane = external global i8*, align 8

define void @FindFreeHorzSeg(i64 %startCol, i64 %row, i64* %rowStart) {
; Ensure that the loop increment basic block is rotated into the tail of the
; body, even though it contains a debug intrinsic call.
; CHECK-LABEL: define void @FindFreeHorzSeg(
; CHECK: %dec = add
; CHECK-NEXT: tail call void @llvm.dbg.value
; CHECK-NEXT: br i1 %tobool, label %for.cond, label %for.end

entry:
  br label %for.cond

for.cond:
  %i.0 = phi i64 [ %startCol, %entry ], [ %dec, %for.inc ]
  %cmp = icmp eq i64 %i.0, 0
  br i1 %cmp, label %for.end, label %for.body

for.body:
  %0 = load i64* @channelColumns, align 8
  %mul = mul i64 %0, %row
  %add = add i64 %mul, %i.0
  %1 = load i8** @horzPlane, align 8
  %arrayidx = getelementptr inbounds i8* %1, i64 %add
  %2 = load i8* %arrayidx, align 1
  %tobool = icmp eq i8 %2, 0
  br i1 %tobool, label %for.inc, label %for.end

for.inc:
  %dec = add i64 %i.0, -1
  tail call void @llvm.dbg.value(metadata !{i64 %dec}, i64 0, metadata !{metadata !"undef"})
  br label %for.cond

for.end:
  %add1 = add i64 %i.0, 1
  store i64 %add1, i64* %rowStart, align 8
  ret void
}

!llvm.module.flags = !{!20}
!llvm.dbg.sp = !{!0}

!0 = metadata !{i32 589870, metadata !18, metadata !1, metadata !"tak", metadata !"tak", metadata !"", i32 32, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i32, i32, i32)* @tak, null, null, null, i32 0} ; [ DW_TAG_subprogram ] [line 32] [def] [scope 0] [tak]
!1 = metadata !{i32 589865, metadata !18} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 589841, metadata !18, i32 12, metadata !"clang version 2.9 (trunk 125492)", i1 true, metadata !"", i32 0, metadata !19, metadata !19, null, null, null, metadata !""} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 589845, metadata !18, metadata !1, metadata !"", i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !4, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 589860, null, metadata !2, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 590081, metadata !0, metadata !"x", metadata !1, i32 32, metadata !5, i32 0} ; [ DW_TAG_arg_variable ]
!7 = metadata !{i32 32, i32 13, metadata !0, null}
!8 = metadata !{i32 590081, metadata !0, metadata !"y", metadata !1, i32 32, metadata !5, i32 0} ; [ DW_TAG_arg_variable ]
!9 = metadata !{i32 32, i32 20, metadata !0, null}
!10 = metadata !{i32 590081, metadata !0, metadata !"z", metadata !1, i32 32, metadata !5, i32 0} ; [ DW_TAG_arg_variable ]
!11 = metadata !{i32 32, i32 27, metadata !0, null}
!12 = metadata !{i32 33, i32 3, metadata !13, null}
!13 = metadata !{i32 589835, metadata !18, metadata !0, i32 32, i32 30, i32 6} ; [ DW_TAG_lexical_block ]
!14 = metadata !{i32 34, i32 5, metadata !15, null}
!15 = metadata !{i32 589835, metadata !18, metadata !13, i32 33, i32 14, i32 7} ; [ DW_TAG_lexical_block ]
!16 = metadata !{i32 36, i32 3, metadata !13, null}
!17 = metadata !{i32 37, i32 1, metadata !13, null}
!18 = metadata !{metadata !"/Volumes/Lalgate/cj/llvm/projects/llvm-test/SingleSource/Benchmarks/BenchmarkGame/recursive.c", metadata !"/Volumes/Lalgate/cj/D/projects/llvm-test/SingleSource/Benchmarks/BenchmarkGame"}
!19 = metadata !{i32 0}
!20 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
