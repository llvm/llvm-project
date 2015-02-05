; RUN: opt %loadPolly -pass-remarks-missed="polly-detect" -polly-detect-track-failures -polly-detect -analyze < %s 2>&1| FileCheck %s

; struct b {
;   double **b;
; };
;
; void a(struct b *A) {
;   for (int i=0; i<32; i++)
;     A->b[i] = 0;
; }

; CHECK: remark: ReportVariantBasePtr01.c:6:8: The following errors keep this region from being a Scop.
; CHECK: remark: ReportVariantBasePtr01.c:7:5: The base address of this array is not invariant inside the loop

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.b = type { double** }

define void @a(%struct.b* nocapture readonly %A) #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  tail call void @llvm.dbg.value(metadata %struct.b* %A, i64 0, metadata !16), !dbg !23
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !17), !dbg !25
  %b = getelementptr inbounds %struct.b* %A, i64 0, i32 0, !dbg !26
  br label %for.body, !dbg !27

for.body:                                         ; preds = %for.body, %entry.split
  %indvar4 = phi i64 [ %indvar.next, %for.body ], [ 0, %entry.split ]
  %0 = mul i64 %indvar4, 4, !dbg !26
  %1 = add i64 %0, 3, !dbg !26
  %2 = add i64 %0, 2, !dbg !26
  %3 = add i64 %0, 1, !dbg !26
  %4 = load double*** %b, align 8, !dbg !26, !tbaa !28
  %arrayidx = getelementptr double** %4, i64 %0, !dbg !26
  store double* null, double** %arrayidx, align 8, !dbg !26, !tbaa !33
  %5 = load double*** %b, align 8, !dbg !26, !tbaa !28
  %arrayidx.1 = getelementptr double** %5, i64 %3, !dbg !26
  store double* null, double** %arrayidx.1, align 8, !dbg !26, !tbaa !33
  %6 = load double*** %b, align 8, !dbg !26, !tbaa !28
  %arrayidx.2 = getelementptr double** %6, i64 %2, !dbg !26
  store double* null, double** %arrayidx.2, align 8, !dbg !26, !tbaa !33
  %7 = load double*** %b, align 8, !dbg !26, !tbaa !28
  %arrayidx.3 = getelementptr double** %7, i64 %1, !dbg !26
  store double* null, double** %arrayidx.3, align 8, !dbg !26, !tbaa !33
  %indvar.next = add i64 %indvar4, 1, !dbg !27
  %exitcond = icmp eq i64 %indvar.next, 8, !dbg !27
  br i1 %exitcond, label %for.end, label %for.body, !dbg !27

for.end:                                          ; preds = %for.body
  ret void, !dbg !34
}

declare void @llvm.dbg.value(metadata, i64, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!20, !21}
!llvm.ident = !{!22}

!0 = !{!"0x11\0012\00clang version 3.5.0 \001\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [test/ScopDetectionDiagnostics/ReportVariantBasePtr01.c] [DW_LANG_C99]
!1 = !{!"ReportVariantBasePtr01.c", !"test/ScopDetectionDiagnostics"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00a\00a\00\005\000\001\000\006\00256\001\005", !1, !5, !6, null, void (%struct.b*)* @a, null, null, !15} ; [ DW_TAG_subprogram ] [line 5] [def] [a]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [test/ScopDetectionDiagnostics/ReportVariantBasePtr01.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null, !8}
!8 = !{!"0xf\00\000\0064\0064\000\000", null, null, !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from b]
!9 = !{!"0x13\00b\001\0064\0064\000\000\000", !1, null, null, !10, null, null, null} ; [ DW_TAG_structure_type ] [b] [line 1, size 64, align 64, offset 0] [def] [from ]
!10 = !{!11}
!11 = !{!"0xd\00b\002\0064\0064\000\000", !1, !9, !12} ; [ DW_TAG_member ] [b] [line 2, size 64, align 64, offset 0] [from ]
!12 = !{!"0xf\00\000\0064\0064\000\000", null, null, !13} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!13 = !{!"0xf\00\000\0064\0064\000\000", null, null, !14} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from double]
!14 = !{!"0x24\00double\000\0064\0064\000\000\004", null, null} ; [ DW_TAG_base_type ] [double] [line 0, size 64, align 64, offset 0, enc DW_ATE_float]
!15 = !{!16, !17}
!16 = !{!"0x101\00A\0016777221\000", !4, !5, !8} ; [ DW_TAG_arg_variable ] [A] [line 5]
!17 = !{!"0x100\00i\006\000", !18, !5, !19} ; [ DW_TAG_auto_variable ] [i] [line 6]
!18 = !{!"0xb\006\003\000", !1, !4} ; [ DW_TAG_lexical_block ] [test/ScopDetectionDiagnostics/ReportVariantBasePtr01.c]
!19 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!20 = !{i32 2, !"Dwarf Version", i32 4}
!21 = !{i32 2, !"Debug Info Version", i32 2}
!22 = !{!"clang version 3.5.0 "}
!23 = !MDLocation(line: 5, column: 18, scope: !4)
!24 = !{i32 0}
!25 = !MDLocation(line: 6, column: 12, scope: !18)
!26 = !MDLocation(line: 7, column: 5, scope: !18)
!27 = !MDLocation(line: 6, column: 8, scope: !18)
!28 = !{!29, !30, i64 0}
!29 = !{!"b", !30, i64 0}
!30 = !{!"any pointer", !31, i64 0}
!31 = !{!"omnipotent char", !32, i64 0}
!32 = !{!"Simple C/C++ TBAA"}
!33 = !{!30, !30, i64 0}
!34 = !MDLocation(line: 8, column: 1, scope: !4)
