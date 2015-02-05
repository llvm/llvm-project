; RUN: opt %loadPolly -pass-remarks-missed="polly-detect" -polly-detect-track-failures -polly-detect -analyze < %s 2>&1| FileCheck %s

; 1 void differenttypes(char *A)
; 2 {
; 3   for (long i = 0; i < 1024; ++i)
; 4     ((float*)A)[i] = ((double*)A)[i];
; 5 }

; CHECK: remark: /tmp/test.c:3:20: The following errors keep this region from being a Scop.
; CHECK-NEXT: remark: /tmp/test.c:4:14: The array "A" is accessed through elements that differ in size
; CHECK-NEXT: remark: /tmp/test.c:4:32: Invalid Scop candidate ends here.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @differenttypes(i8* nocapture %A)  {
entry:
  br label %for.body, !dbg !10

for.body:                                         ; preds = %for.body, %entry
  %i.05 = phi i64 [ 0, %entry ], [ %tmp11, %for.body ]
  %tmp = shl i64 %i.05, 3, !dbg !15
  %uglygep = getelementptr i8* %A, i64 %tmp
  %arrayidx = bitcast i8* %uglygep to double*, !dbg !16
  %tmp9 = shl i64 %i.05, 2, !dbg !15
  %uglygep7 = getelementptr i8* %A, i64 %tmp9
  %arrayidx1 = bitcast i8* %uglygep7 to float*, !dbg !17
  %tmp10 = load double* %arrayidx, align 8, !dbg !16, !tbaa !18
  %conv = fptrunc double %tmp10 to float, !dbg !16
  store float %conv, float* %arrayidx1, align 4, !dbg !17, !tbaa !22
  %tmp11 = add nsw i64 %i.05, 1, !dbg !24
  %exitcond = icmp eq i64 %tmp11, 1024, !dbg !10
  br i1 %exitcond, label %for.end, label %for.body, !dbg !10

for.end:                                          ; preds = %for.body
  ret void, !dbg !25
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = !{!"0x11\0012\00clang version 3.6.0 \001\00\000\00\002", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/home/grosser/Projects/polly/git/tools/polly/test/ScopDetectionDiagnostics//tmp/test.c] [DW_LANG_C99]
!1 = !{!"/tmp/test.c", !"/home/grosser/Projects/polly/git/tools/polly/test/ScopDetectionDiagnostics"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00differenttypes\00differenttypes\00\001\000\001\000\006\00256\001\002", !1, !5, !6, null, void (i8*)* @differenttypes, null, null, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [scope 2] [differenttypes]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/home/grosser/Projects/polly/git/tools/polly/test/ScopDetectionDiagnostics//tmp/test.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !2, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 2}
!9 = !{!"clang version 3.6.0 "}
!10 = !MDLocation(line: 3, column: 20, scope: !11)
!11 = !{!"0xb\002", !1, !12} ; [ DW_TAG_lexical_block ] [/home/grosser/Projects/polly/git/tools/polly/test/ScopDetectionDiagnostics//tmp/test.c]
!12 = !{!"0xb\001", !1, !13} ; [ DW_TAG_lexical_block ] [/home/grosser/Projects/polly/git/tools/polly/test/ScopDetectionDiagnostics//tmp/test.c]
!13 = !{!"0xb\003\003\001", !1, !14} ; [ DW_TAG_lexical_block ] [/home/grosser/Projects/polly/git/tools/polly/test/ScopDetectionDiagnostics//tmp/test.c]
!14 = !{!"0xb\003\003\000", !1, !4} ; [ DW_TAG_lexical_block ] [/home/grosser/Projects/polly/git/tools/polly/test/ScopDetectionDiagnostics//tmp/test.c]
!15 = !MDLocation(line: 4, column: 32, scope: !13)
!16 = !MDLocation(line: 4, column: 22, scope: !13)
!17 = !MDLocation(line: 4, column: 14, scope: !13)
!18 = !{!19, !19, i64 0}
!19 = !{!"double", !20, i64 0}
!20 = !{!"omnipotent char", !21, i64 0}
!21 = !{!"Simple C/C++ TBAA"}
!22 = !{!23, !23, i64 0}
!23 = !{!"float", !20, i64 0}
!24 = !MDLocation(line: 3, column: 30, scope: !13)
!25 = !MDLocation(line: 5, column: 1, scope: !4)
