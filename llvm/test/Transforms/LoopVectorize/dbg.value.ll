; RUN: opt < %s -S -loop-vectorize -force-vector-unroll=1 -force-vector-width=4 -dce -instcombine | FileCheck %s
; Make sure we vectorize with debugging turned on.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

@A = global [1024 x i32] zeroinitializer, align 16
@B = global [1024 x i32] zeroinitializer, align 16
@C = global [1024 x i32] zeroinitializer, align 16

; CHECK-LABEL: @test(
define i32 @test() #0 {
entry:
  tail call void @llvm.dbg.value(metadata !1, i64 0, metadata !9), !dbg !18
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
  tail call void @llvm.dbg.value(metadata !{null}, i64 0, metadata !9), !dbg !18
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !18
  %exitcond = icmp ne i32 %lftr.wideiv, 1024, !dbg !18
  br i1 %exitcond, label %for.body, label %for.end, !dbg !18

for.end:
  ret i32 0, !dbg !24
}

declare void @llvm.dbg.declare(metadata, metadata) #1

declare void @llvm.dbg.value(metadata, i64, metadata) #1

attributes #0 = { nounwind ssp uwtable "fp-contract-model"="standard" "no-frame-pointer-elim" "no-frame-pointer-elim-non-leaf" "relocation-model"="pic" "ssp-buffers-size"="8" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!26}

!0 = metadata !{i32 786449, metadata !25, i32 4, metadata !"clang", i1 true, metadata !"", i32 0, metadata !1, metadata !1, metadata !2, metadata !11, null, metadata !""}
!1 = metadata !{i32 0}
!2 = metadata !{metadata !3}
!3 = metadata !{i32 786478, metadata !25, metadata !4, metadata !"test", metadata !"test", metadata !"test", i32 5, metadata !5, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 ()* @test, null, null, metadata !8, i32 5}
!4 = metadata !{i32 786473, metadata !25}
!5 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !6, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!6 = metadata !{metadata !7}
!7 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5}
!8 = metadata !{metadata !9}
!9 = metadata !{i32 786688, metadata !10, metadata !"i", metadata !4, i32 6, metadata !7, i32 0, i32 0}
!10 = metadata !{i32 786443, metadata !25, metadata !3, i32 6, i32 0, i32 0}
!11 = metadata !{metadata !12, metadata !16, metadata !17}
!12 = metadata !{i32 786484, i32 0, null, metadata !"A", metadata !"A", metadata !"", metadata !4, i32 1, metadata !13, i32 0, i32 1, [1024 x i32]* @A, null}
!13 = metadata !{i32 786433, null, null, null, i32 0, i64 32768, i64 32, i32 0, i32 0, metadata !7, metadata !14, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 32768, align 32, offset 0] [from int]
!14 = metadata !{metadata !15}
!15 = metadata !{i32 786465, i64 0, i64 1024}
!16 = metadata !{i32 786484, i32 0, null, metadata !"B", metadata !"B", metadata !"", metadata !4, i32 2, metadata !13, i32 0, i32 1, [1024 x i32]* @B, null}
!17 = metadata !{i32 786484, i32 0, null, metadata !"C", metadata !"C", metadata !"", metadata !4, i32 3, metadata !13, i32 0, i32 1, [1024 x i32]* @C, null} 
!18 = metadata !{i32 6, i32 0, metadata !10, null}
!19 = metadata !{i32 7, i32 0, metadata !20, null}
!20 = metadata !{i32 786443, metadata !25, metadata !10, i32 6, i32 0, i32 1}
!24 = metadata !{i32 9, i32 0, metadata !3, null}
!25 = metadata !{metadata !"test", metadata !"/path/to/somewhere"}
!26 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
