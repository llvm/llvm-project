; RUN: opt < %s -S -loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -dce -instcombine | FileCheck %s
; Make sure we vectorize with debugging turned on.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

@A = global [1024 x i32] zeroinitializer, align 16
@B = global [1024 x i32] zeroinitializer, align 16
@C = global [1024 x i32] zeroinitializer, align 16

; CHECK-LABEL: @test(
define i32 @test() #0 {
entry:
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !9, metadata !{}), !dbg !18
  br label %for.body, !dbg !18

for.body:
  ;CHECK: load <4 x i32>
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32]* @B, i64 0, i64 %indvars.iv, !dbg !19
  %0 = load i32* %arrayidx, align 4, !dbg !19
  %arrayidx2 = getelementptr inbounds [1024 x i32]* @C, i64 0, i64 %indvars.iv, !dbg !19
  %1 = load i32* %arrayidx2, align 4, !dbg !19
  %add = add nsw i32 %1, %0, !dbg !19
  %arrayidx4 = getelementptr inbounds [1024 x i32]* @A, i64 0, i64 %indvars.iv, !dbg !19
  store i32 %add, i32* %arrayidx4, align 4, !dbg !19
  %indvars.iv.next = add i64 %indvars.iv, 1, !dbg !18
  tail call void @llvm.dbg.value(metadata !{null}, i64 0, metadata !9, metadata !{}), !dbg !18
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !18
  %exitcond = icmp ne i32 %lftr.wideiv, 1024, !dbg !18
  br i1 %exitcond, label %for.body, label %for.end, !dbg !18

for.end:
  ret i32 0, !dbg !24
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable "fp-contract-model"="standard" "no-frame-pointer-elim" "no-frame-pointer-elim-non-leaf" "relocation-model"="pic" "ssp-buffers-size"="8" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!26}

!0 = !{!"0x11\004\00clang\001\00\000\00\000", !25, !1, !1, !2, !11, null} ; [ DW_TAG_compile_unit ]
!1 = !{i32 0}
!2 = !{!3}
!3 = !{!"0x2e\00test\00test\00test\005\000\001\000\006\00256\001\005", !25, !4, !5, null, i32 ()* @test, null, null, !8} ; [ DW_TAG_subprogram ]
!4 = !{!"0x29", !25} ; [ DW_TAG_file_type ]
!5 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !6, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!6 = !{!7}
!7 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!8 = !{!9}
!9 = !{!"0x100\00i\006\000", !10, !4, !7} ; [ DW_TAG_auto_variable ]
!10 = !{!"0xb\006\000\000", !25, !3} ; [ DW_TAG_lexical_block ]
!11 = !{!12, !16, !17}
!12 = !{!"0x34\00A\00A\00\001\000\001", null, !4, !13, [1024 x i32]* @A, null} ; [ DW_TAG_variable ]
!13 = !{!"0x1\00\000\0032768\0032\000\000", null, null, !7, !14, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 32768, align 32, offset 0] [from int]
!14 = !{!15}
!15 = !{i32 786465, i64 0, i64 1024}
!16 = !{!"0x34\00B\00B\00\002\000\001", null, !4, !13, [1024 x i32]* @B, null} ; [ DW_TAG_variable ]
!17 = !{!"0x34\00C\00C\00\003\000\001", null, !4, !13, [1024 x i32]* @C, null} ; [ DW_TAG_variable ]
!18 = !MDLocation(line: 6, scope: !10)
!19 = !MDLocation(line: 7, scope: !20)
!20 = !{!"0xb\006\000\001", !25, !10} ; [ DW_TAG_lexical_block ]
!24 = !MDLocation(line: 9, scope: !3)
!25 = !{!"test", !"/path/to/somewhere"}
!26 = !{i32 1, !"Debug Info Version", i32 2}
