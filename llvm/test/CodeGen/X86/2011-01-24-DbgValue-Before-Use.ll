; RUN: llc < %s | FileCheck %s
; RUN: llc < %s -regalloc=basic | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

; Check debug info for variable z_s
;CHECK: .long Lset13
;CHECK-NEXT:  ## DW_AT_decl_file
;CHECK-NEXT:  ## DW_AT_decl_line
;CHECK-NEXT:  ## DW_AT_type
;CHECK-NEXT:  ## DW_AT_location


@.str1 = private unnamed_addr constant [14 x i8] c"m=%u, z_s=%d\0A\00"
@str = internal constant [21 x i8] c"Failing test vector:\00"

define i64 @gcd(i64 %a, i64 %b) nounwind readnone optsize noinline ssp {
entry:
  tail call void @llvm.dbg.value(metadata !{i64 %a}, i64 0, metadata !10), !dbg !18
  tail call void @llvm.dbg.value(metadata !{i64 %b}, i64 0, metadata !11), !dbg !19
  br label %while.body, !dbg !20

while.body:                                       ; preds = %while.body, %entry
  %b.addr.0 = phi i64 [ %b, %entry ], [ %rem, %while.body ]
  %a.addr.0 = phi i64 [ %a, %entry ], [ %b.addr.0, %while.body ]
  %rem = srem i64 %a.addr.0, %b.addr.0, !dbg !21
  %cmp = icmp eq i64 %rem, 0, !dbg !23
  br i1 %cmp, label %if.then, label %while.body, !dbg !23

if.then:                                          ; preds = %while.body
  tail call void @llvm.dbg.value(metadata !{i64 %rem}, i64 0, metadata !12), !dbg !21
  ret i64 %b.addr.0, !dbg !23
}

define i32 @main() nounwind optsize ssp {
entry:
  %call = tail call i32 @rand() nounwind optsize, !dbg !24
  tail call void @llvm.dbg.value(metadata !{i32 %call}, i64 0, metadata !14), !dbg !24
  %cmp = icmp ugt i32 %call, 21, !dbg !25
  br i1 %cmp, label %cond.true, label %cond.end, !dbg !25

cond.true:                                        ; preds = %entry
  %call1 = tail call i32 @rand() nounwind optsize, !dbg !25
  br label %cond.end, !dbg !25

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i32 [ %call1, %cond.true ], [ %call, %entry ], !dbg !25
  tail call void @llvm.dbg.value(metadata !{i32 %cond}, i64 0, metadata !17), !dbg !25
  %conv = sext i32 %cond to i64, !dbg !26
  %conv5 = zext i32 %call to i64, !dbg !26
  %call6 = tail call i64 @gcd(i64 %conv, i64 %conv5) optsize, !dbg !26
  %cmp7 = icmp eq i64 %call6, 0
  br i1 %cmp7, label %return, label %if.then, !dbg !26

if.then:                                          ; preds = %cond.end
  %puts = tail call i32 @puts(i8* getelementptr inbounds ([21 x i8]* @str, i64 0, i64 0))
  %call12 = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([14 x i8]* @.str1, i64 0, i64 0), i32 %call, i32 %cond) nounwind optsize, !dbg !26
  ret i32 1, !dbg !27

return:                                           ; preds = %cond.end
  ret i32 0, !dbg !27
}

declare i32 @rand() optsize

declare i32 @printf(i8* nocapture, ...) nounwind optsize

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

declare i32 @puts(i8* nocapture) nounwind

!llvm.dbg.sp = !{!0, !6}
!llvm.dbg.lv.gcd = !{!10, !11, !12}
!llvm.dbg.lv.main = !{!14, !17}

!0 = metadata !{i32 589870, i32 0, metadata !1, metadata !"gcd", metadata !"gcd", metadata !"", metadata !1, i32 5, metadata !3, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 true, i64 (i64, i64)* @gcd} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 589865, metadata !"rem_small.c", metadata !"/private/tmp", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 589841, i32 0, i32 12, metadata !"rem_small.c", metadata !"/private/tmp", metadata !"clang version 2.9 (trunk 124117)", i1 true, i1 true, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 589845, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !4, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 589860, metadata !2, metadata !"long int", null, i32 0, i64 64, i64 64, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 589870, i32 0, metadata !1, metadata !"main", metadata !"main", metadata !"", metadata !1, i32 25, metadata !7, i1 false, i1 true, i32 0, i32 0, i32 0, i32 0, i1 true, i32 ()* @main} ; [ DW_TAG_subprogram ]
!7 = metadata !{i32 589845, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !8, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!8 = metadata !{metadata !9}
!9 = metadata !{i32 589860, metadata !2, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!10 = metadata !{i32 590081, metadata !0, metadata !"a", metadata !1, i32 5, metadata !5, i32 0} ; [ DW_TAG_arg_variable ]
!11 = metadata !{i32 590081, metadata !0, metadata !"b", metadata !1, i32 5, metadata !5, i32 0} ; [ DW_TAG_arg_variable ]
!12 = metadata !{i32 590080, metadata !13, metadata !"c", metadata !1, i32 6, metadata !5, i32 0} ; [ DW_TAG_auto_variable ]
!13 = metadata !{i32 589835, metadata !0, i32 5, i32 52, metadata !1, i32 0} ; [ DW_TAG_lexical_block ]
!14 = metadata !{i32 590080, metadata !15, metadata !"m", metadata !1, i32 26, metadata !16, i32 0} ; [ DW_TAG_auto_variable ]
!15 = metadata !{i32 589835, metadata !6, i32 25, i32 12, metadata !1, i32 2} ; [ DW_TAG_lexical_block ]
!16 = metadata !{i32 589860, metadata !2, metadata !"unsigned int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!17 = metadata !{i32 590080, metadata !15, metadata !"z_s", metadata !1, i32 27, metadata !9, i32 0} ; [ DW_TAG_auto_variable ]
!18 = metadata !{i32 5, i32 41, metadata !0, null}
!19 = metadata !{i32 5, i32 49, metadata !0, null}
!20 = metadata !{i32 7, i32 5, metadata !13, null}
!21 = metadata !{i32 8, i32 9, metadata !22, null}
!22 = metadata !{i32 589835, metadata !13, i32 7, i32 14, metadata !1, i32 1} ; [ DW_TAG_lexical_block ]
!23 = metadata !{i32 9, i32 9, metadata !22, null}
!24 = metadata !{i32 26, i32 38, metadata !15, null}
!25 = metadata !{i32 27, i32 38, metadata !15, null}
!26 = metadata !{i32 28, i32 9, metadata !15, null}
!27 = metadata !{i32 30, i32 1, metadata !15, null}
