; RUN: llc < %s - | FileCheck %s
; Radar 9376013
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-macosx10.6.7"

;CHECK: Ldebug_loc0:
;CHECK-NEXT:        .long   Ltmp0
;CHECK-NEXT:        .long   Ltmp1
;CHECK-NEXT: Lset[[N:[0-9]+]] = Ltmp{{[0-9]+}}-Ltmp[[M:[0-9]+]]        @ Loc expr size
;CHECK-NEXT:        .short  Lset[[N]]
;CHECK-NEXT: Ltmp[[M]]:
;CHECK-NEXT:        .byte   144                     @ DW_OP_regx for S register

define void @_Z3foov() optsize ssp {
entry:
  %call = tail call float @_Z3barv() optsize, !dbg !11
  tail call void @llvm.dbg.value(metadata !{float %call}, i64 0, metadata !5), !dbg !11
  %call16 = tail call float @_Z2f2v() optsize, !dbg !12
  %cmp7 = fcmp olt float %call, %call16, !dbg !12
  br i1 %cmp7, label %for.body, label %for.end, !dbg !12

for.body:                                         ; preds = %entry, %for.body
  %k.08 = phi float [ %inc, %for.body ], [ %call, %entry ]
  %call4 = tail call float @_Z2f3f(float %k.08) optsize, !dbg !13
  %inc = fadd float %k.08, 1.000000e+00, !dbg !14
  %call1 = tail call float @_Z2f2v() optsize, !dbg !12
  %cmp = fcmp olt float %inc, %call1, !dbg !12
  br i1 %cmp, label %for.body, label %for.end, !dbg !12

for.end:                                          ; preds = %for.body, %entry
  ret void, !dbg !15
}

declare float @_Z3barv() optsize

declare float @_Z2f2v() optsize

declare float @_Z2f3f(float) optsize

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.dbg.sp = !{!1}
!llvm.dbg.lv._Z3foov = !{!5, !8}

!0 = metadata !{i32 589841, i32 0, i32 4, metadata !"k.cc", metadata !"/private/tmp", metadata !"clang version 3.0 (trunk 130845)", i1 true, i1 true, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 589870, i32 0, metadata !2, metadata !"foo", metadata !"foo", metadata !"_Z3foov", metadata !2, i32 5, metadata !3, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 true, void ()* @_Z3foov, null, null} ; [ DW_TAG_subprogram ]
!2 = metadata !{i32 589865, metadata !"k.cc", metadata !"/private/tmp", metadata !0} ; [ DW_TAG_file_type ]
!3 = metadata !{i32 589845, metadata !2, metadata !"", metadata !2, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !4, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{null}
!5 = metadata !{i32 590080, metadata !6, metadata !"k", metadata !2, i32 6, metadata !7, i32 0} ; [ DW_TAG_auto_variable ]
!6 = metadata !{i32 589835, metadata !1, i32 5, i32 12, metadata !2, i32 0} ; [ DW_TAG_lexical_block ]
!7 = metadata !{i32 589860, metadata !0, metadata !"float", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 4} ; [ DW_TAG_base_type ]
!8 = metadata !{i32 590080, metadata !9, metadata !"y", metadata !2, i32 8, metadata !7, i32 0} ; [ DW_TAG_auto_variable ]
!9 = metadata !{i32 589835, metadata !10, i32 7, i32 25, metadata !2, i32 2} ; [ DW_TAG_lexical_block ]
!10 = metadata !{i32 589835, metadata !6, i32 7, i32 3, metadata !2, i32 1} ; [ DW_TAG_lexical_block ]
!11 = metadata !{i32 6, i32 18, metadata !6, null}
!12 = metadata !{i32 7, i32 3, metadata !6, null}
!13 = metadata !{i32 8, i32 20, metadata !9, null}
!14 = metadata !{i32 7, i32 20, metadata !10, null}
!15 = metadata !{i32 10, i32 1, metadata !6, null}
