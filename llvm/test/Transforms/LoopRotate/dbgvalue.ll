; RUN: opt -S -loop-rotate < %s | FileCheck %s

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

define i32 @tak(i32 %x, i32 %y, i32 %z) nounwind ssp {
; CHECK-LABEL: define i32 @tak(
; CHECK: entry
; CHECK-NEXT: call void @llvm.dbg.value(metadata i32 %x

entry:
  br label %tailrecurse

tailrecurse:                                      ; preds = %if.then, %entry
  %x.tr = phi i32 [ %x, %entry ], [ %call, %if.then ]
  %y.tr = phi i32 [ %y, %entry ], [ %call9, %if.then ]
  %z.tr = phi i32 [ %z, %entry ], [ %call14, %if.then ]
  tail call void @llvm.dbg.value(metadata i32 %x.tr, i64 0, metadata !6, metadata !{}), !dbg !7
  tail call void @llvm.dbg.value(metadata i32 %y.tr, i64 0, metadata !8, metadata !{}), !dbg !9
  tail call void @llvm.dbg.value(metadata i32 %z.tr, i64 0, metadata !10, metadata !{}), !dbg !11
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
; CHECK: %cmp = icmp
; CHECK: br i1 %cmp
; CHECK: phi i64 [ %{{[^,]*}}, %{{[^,]*}} ]
; CHECK-NEXT: br label %for.end


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
  tail call void @llvm.dbg.value(metadata i64 %dec, i64 0, metadata !{!"undef"}, metadata !{})
  br label %for.cond

for.end:
  %add1 = add i64 %i.0, 1
  store i64 %add1, i64* %rowStart, align 8
  ret void
}

!llvm.module.flags = !{!20}
!llvm.dbg.sp = !{!0}

!0 = !{!"0x2e\00tak\00tak\00\0032\000\001\000\006\00256\000\000", !18, !1, !3, null, i32 (i32, i32, i32)* @tak, null, null, null} ; [ DW_TAG_subprogram ] [line 32] [def] [scope 0] [tak]
!1 = !{!"0x29", !18} ; [ DW_TAG_file_type ]
!2 = !{!"0x11\0012\00clang version 2.9 (trunk 125492)\001\00\000\00\000", !18, !19, !19, null, null, null} ; [ DW_TAG_compile_unit ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !18, !1, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{!5}
!5 = !{!"0x24\00int\000\0032\0032\000\000\005", null, !2} ; [ DW_TAG_base_type ]
!6 = !{!"0x101\00x\0032\000", !0, !1, !5} ; [ DW_TAG_arg_variable ]
!7 = !MDLocation(line: 32, column: 13, scope: !0)
!8 = !{!"0x101\00y\0032\000", !0, !1, !5} ; [ DW_TAG_arg_variable ]
!9 = !MDLocation(line: 32, column: 20, scope: !0)
!10 = !{!"0x101\00z\0032\000", !0, !1, !5} ; [ DW_TAG_arg_variable ]
!11 = !MDLocation(line: 32, column: 27, scope: !0)
!12 = !MDLocation(line: 33, column: 3, scope: !13)
!13 = !{!"0xb\0032\0030\006", !18, !0} ; [ DW_TAG_lexical_block ]
!14 = !MDLocation(line: 34, column: 5, scope: !15)
!15 = !{!"0xb\0033\0014\007", !18, !13} ; [ DW_TAG_lexical_block ]
!16 = !MDLocation(line: 36, column: 3, scope: !13)
!17 = !MDLocation(line: 37, column: 1, scope: !13)
!18 = !{!"/Volumes/Lalgate/cj/llvm/projects/llvm-test/SingleSource/Benchmarks/BenchmarkGame/recursive.c", !"/Volumes/Lalgate/cj/D/projects/llvm-test/SingleSource/Benchmarks/BenchmarkGame"}
!19 = !{i32 0}
!20 = !{i32 1, !"Debug Info Version", i32 2}
