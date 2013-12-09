; RUN: llvm-jitlistener %s | FileCheck %s

; CHECK: Method load [1]: main, Size = 164
; CHECK: Method unload [1]

; ModuleID = '<stdin>'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@zero_int = common global i32 0, align 4
@zero_arr = common global [10 x i32] zeroinitializer, align 16
@zero_double = common global double 0.000000e+00, align 8

define i32 @main() nounwind uwtable {
entry:
  %retval = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 0, i32* %retval
  %0 = load i32* @zero_int, align 4, !dbg !21
  %add = add nsw i32 %0, 5, !dbg !21
  %idxprom = sext i32 %add to i64, !dbg !21
  %arrayidx = getelementptr inbounds [10 x i32]* @zero_arr, i32 0, i64 %idxprom, !dbg !21
  store i32 40, i32* %arrayidx, align 4, !dbg !21
  %1 = load double* @zero_double, align 8, !dbg !23
  %cmp = fcmp olt double %1, 1.000000e+00, !dbg !23
  br i1 %cmp, label %if.then, label %if.end, !dbg !23

if.then:                                          ; preds = %entry
  %2 = load i32* @zero_int, align 4, !dbg !24
  %add1 = add nsw i32 %2, 2, !dbg !24
  %idxprom2 = sext i32 %add1 to i64, !dbg !24
  %arrayidx3 = getelementptr inbounds [10 x i32]* @zero_arr, i32 0, i64 %idxprom2, !dbg !24
  store i32 70, i32* %arrayidx3, align 4, !dbg !24
  br label %if.end, !dbg !24

if.end:                                           ; preds = %if.then, %entry
  call void @llvm.dbg.declare(metadata !{i32* %i}, metadata !25), !dbg !27
  store i32 1, i32* %i, align 4, !dbg !28
  br label %for.cond, !dbg !28

for.cond:                                         ; preds = %for.inc, %if.end
  %3 = load i32* %i, align 4, !dbg !28
  %cmp4 = icmp slt i32 %3, 10, !dbg !28
  br i1 %cmp4, label %for.body, label %for.end, !dbg !28

for.body:                                         ; preds = %for.cond
  %4 = load i32* %i, align 4, !dbg !29
  %sub = sub nsw i32 %4, 1, !dbg !29
  %idxprom5 = sext i32 %sub to i64, !dbg !29
  %arrayidx6 = getelementptr inbounds [10 x i32]* @zero_arr, i32 0, i64 %idxprom5, !dbg !29
  %5 = load i32* %arrayidx6, align 4, !dbg !29
  %6 = load i32* %i, align 4, !dbg !29
  %idxprom7 = sext i32 %6 to i64, !dbg !29
  %arrayidx8 = getelementptr inbounds [10 x i32]* @zero_arr, i32 0, i64 %idxprom7, !dbg !29
  %7 = load i32* %arrayidx8, align 4, !dbg !29
  %add9 = add nsw i32 %5, %7, !dbg !29
  %8 = load i32* %i, align 4, !dbg !29
  %idxprom10 = sext i32 %8 to i64, !dbg !29
  %arrayidx11 = getelementptr inbounds [10 x i32]* @zero_arr, i32 0, i64 %idxprom10, !dbg !29
  store i32 %add9, i32* %arrayidx11, align 4, !dbg !29
  br label %for.inc, !dbg !31

for.inc:                                          ; preds = %for.body
  %9 = load i32* %i, align 4, !dbg !32
  %inc = add nsw i32 %9, 1, !dbg !32
  store i32 %inc, i32* %i, align 4, !dbg !32
  br label %for.cond, !dbg !32

for.end:                                          ; preds = %for.cond
  %10 = load i32* getelementptr inbounds ([10 x i32]* @zero_arr, i32 0, i64 9), align 4, !dbg !33
  %cmp12 = icmp eq i32 %10, 110, !dbg !33
  %cond = select i1 %cmp12, i32 0, i32 -1, !dbg !33
  ret i32 %cond, !dbg !33
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!35}

!0 = metadata !{i32 720913, metadata !34, i32 12, metadata !"clang version 3.1 ()", i1 true, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !12, null, metadata !""} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{i32 720942, metadata !34, metadata !6, metadata !"main", metadata !"main", metadata !"", i32 6, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, i32 ()* @main, null, null, metadata !10, i32 0} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 720937, metadata !34} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 720917, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9}
!9 = metadata !{i32 720932, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!10 = metadata !{metadata !11}
!11 = metadata !{i32 720932}                      ; [ DW_TAG_base_type ]
!12 = metadata !{metadata !14, metadata !15, metadata !17}
!14 = metadata !{i32 720948, i32 0, null, metadata !"zero_int", metadata !"zero_int", metadata !"", metadata !6, i32 1, metadata !9, i32 0, i32 1, i32* @zero_int, null} ; [ DW_TAG_variable ]
!15 = metadata !{i32 720948, i32 0, null, metadata !"zero_double", metadata !"zero_double", metadata !"", metadata !6, i32 2, metadata !16, i32 0, i32 1, double* @zero_double, null} ; [ DW_TAG_variable ]
!16 = metadata !{i32 720932, null, null, metadata !"double", i32 0, i64 64, i64 64, i64 0, i32 0, i32 4} ; [ DW_TAG_base_type ]
!17 = metadata !{i32 720948, i32 0, null, metadata !"zero_arr", metadata !"zero_arr", metadata !"", metadata !6, i32 3, metadata !18, i32 0, i32 1, [10 x i32]* @zero_arr, null} ; [ DW_TAG_variable ]
!18 = metadata !{i32 720897, null, metadata !"", null, i32 0, i64 320, i64 32, i32 0, i32 0, metadata !9, metadata !19, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 320, align 32, offset 0] [from int]
!19 = metadata !{metadata !20}
!20 = metadata !{i32 720929, i64 0, i64 10}        ; [ DW_TAG_subrange_type ]
!21 = metadata !{i32 7, i32 5, metadata !22, null}
!22 = metadata !{i32 720907, metadata !34, metadata !5, i32 6, i32 1, i32 0} ; [ DW_TAG_lexical_block ]
!23 = metadata !{i32 9, i32 5, metadata !22, null}
!24 = metadata !{i32 10, i32 9, metadata !22, null}
!25 = metadata !{i32 721152, metadata !26, metadata !"i", metadata !6, i32 12, metadata !9, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!26 = metadata !{i32 720907, metadata !34, metadata !22, i32 12, i32 5, i32 1} ; [ DW_TAG_lexical_block ]
!27 = metadata !{i32 12, i32 14, metadata !26, null}
!28 = metadata !{i32 12, i32 19, metadata !26, null}
!29 = metadata !{i32 13, i32 9, metadata !30, null}
!30 = metadata !{i32 720907, metadata !34, metadata !26, i32 12, i32 34, i32 2} ; [ DW_TAG_lexical_block ]
!31 = metadata !{i32 14, i32 5, metadata !30, null}
!32 = metadata !{i32 12, i32 29, metadata !26, null}
!33 = metadata !{i32 15, i32 5, metadata !22, null}
!34 = metadata !{metadata !"test-common-symbols.c", metadata !"/store/store/llvm/build"}
!35 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
