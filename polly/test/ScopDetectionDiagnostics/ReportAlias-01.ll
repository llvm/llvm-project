; RUN: opt %loadPolly -polly-use-runtime-alias-checks=false -pass-remarks-missed="polly-detect" -polly-detect-track-failures -polly-detect -analyze < %s 2>&1| FileCheck %s

;void f(int A[], int B[]) {
;  for (int i=0; i<42; i++)
;    A[i] = B[i];
;}

; CHECK: remark: ReportAlias-01.c:2:8: The following errors keep this region from being a Scop.
; CHECK: remark: ReportAlias-01.c:3:5: Accesses to the arrays "B", "A" may access the same memory.
; CHECK: remark: ReportAlias-01.c:3:5: Invalid Scop candidate ends here.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @f(i32* %A, i32* %B) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  tail call void @llvm.dbg.value(metadata i32* %A, i64 0, metadata !13), !dbg !14
  tail call void @llvm.dbg.value(metadata i32* %B, i64 0, metadata !15), !dbg !16
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !18), !dbg !20
  br label %for.body, !dbg !21

for.body:                                         ; preds = %entry.split, %for.body
  %indvar = phi i64 [ 0, %entry.split ], [ %indvar.next, %for.body ]
  %arrayidx = getelementptr i32* %B, i64 %indvar, !dbg !22
  %arrayidx2 = getelementptr i32* %A, i64 %indvar, !dbg !22
  %0 = load i32* %arrayidx, align 4, !dbg !22
  store i32 %0, i32* %arrayidx2, align 4, !dbg !22
  tail call void @llvm.dbg.value(metadata !{null}, i64 0, metadata !18), !dbg !20
  %indvar.next = add i64 %indvar, 1, !dbg !21
  %exitcond = icmp ne i64 %indvar.next, 42, !dbg !21
  br i1 %exitcond, label %for.body, label %for.end, !dbg !21

for.end:                                          ; preds = %for.body
  ret void, !dbg !23
}

declare void @llvm.dbg.declare(metadata, metadata)
declare void @llvm.dbg.value(metadata, i64, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}
!llvm.ident = !{!12}

!0 = !{!"0x11\0012\00clang version 3.6.0 \000\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [test/ScopDetectionDiagnostic/ReportAlias-01.c] [DW_LANG_C99]
!1 = !{!"ReportAlias-01.c", !"test/ScopDetectionDiagnostic/"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00f\00f\00\001\000\001\000\006\00256\000\001", !1, !5, !6, null, void (i32*, i32*)* @f, null, null, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [f]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [test/ScopDetectionDiagnostic/ReportAlias-01.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null, !8, !8}
!8 = !{!"0xf\00\000\0064\0064\000\000", null, null, !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 2}
!12 = !{!"clang version 3.6.0 "}
!13 = !{!"0x101\00A\0016777217\000", !4, !5, !8} ; [ DW_TAG_arg_variable ] [A] [line 1]
!14 = !MDLocation(line: 1, column: 12, scope: !4)
!15 = !{!"0x101\00B\0033554433\000", !4, !5, !8} ; [ DW_TAG_arg_variable ] [B] [line 1]
!16 = !MDLocation(line: 1, column: 21, scope: !4)
!17 = !{i32 0}
!18 = !{!"0x100\00i\002\000", !19, !5, !9} ; [ DW_TAG_auto_variable ] [i] [line 2]
!19 = !{!"0xb\002\003\000", !1, !4} ; [ DW_TAG_lexical_block ] [test/ScopDetectionDiagnostic/ReportAlias-01.c]
!20 = !MDLocation(line: 2, column: 12, scope: !19)
!21 = !MDLocation(line: 2, column: 8, scope: !19)
!22 = !MDLocation(line: 3, column: 5, scope: !19)
!23 = !MDLocation(line: 4, column: 1, scope: !4)
