; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.7.0"

; int depth(double *A, int m) {
;   double y0 = 0; double y1 = 1;
;   for (int i=0; i < m; i++) {
;     y0 = A[4];
;     y1 = A[5];
;   }
;   A[8] = y0; A[8+1] = y1;
; }

;CHECK: @depth
;CHECK: getelementptr inbounds {{.*}}, !dbg ![[LOC:[0-9]+]]
;CHECK: bitcast double* {{.*}}, !dbg ![[LOC]]
;CHECK: load <2 x double>* {{.*}}, !dbg ![[LOC]]
;CHECK: store <2 x double> {{.*}}, !dbg ![[LOC2:[0-9]+]]
;CHECK: ret
;CHECK: ![[LOC]] = !MDLocation(line: 4, scope:
;CHECK: ![[LOC2]] = !MDLocation(line: 7, scope:

define i32 @depth(double* nocapture %A, i32 %m) #0 {
entry:
  tail call void @llvm.dbg.value(metadata double* %A, i64 0, metadata !12, metadata !{}), !dbg !19
  tail call void @llvm.dbg.value(metadata i32 %m, i64 0, metadata !13, metadata !{}), !dbg !19
  tail call void @llvm.dbg.value(metadata i32 00, i64 0, metadata !14, metadata !{}), !dbg !21
  tail call void @llvm.dbg.value(metadata i32 02, i64 0, metadata !15, metadata !{}), !dbg !21
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !16, metadata !{}), !dbg !23
  %cmp8 = icmp sgt i32 %m, 0, !dbg !23
  br i1 %cmp8, label %for.body.lr.ph, label %for.end, !dbg !23

for.body.lr.ph:                                   ; preds = %entry
  %arrayidx = getelementptr inbounds double* %A, i64 4, !dbg !24
  %0 = load double* %arrayidx, align 8, !dbg !24
  %arrayidx1 = getelementptr inbounds double* %A, i64 5, !dbg !29
  %1 = load double* %arrayidx1, align 8, !dbg !29
  br label %for.end, !dbg !23

for.end:                                          ; preds = %for.body.lr.ph, %entry
  %y1.0.lcssa = phi double [ %1, %for.body.lr.ph ], [ 1.000000e+00, %entry ]
  %y0.0.lcssa = phi double [ %0, %for.body.lr.ph ], [ 0.000000e+00, %entry ]
  %arrayidx2 = getelementptr inbounds double* %A, i64 8, !dbg !30
  store double %y0.0.lcssa, double* %arrayidx2, align 8, !dbg !30
  %arrayidx3 = getelementptr inbounds double* %A, i64 9, !dbg !30
  store double %y1.0.lcssa, double* %arrayidx3, align 8, !dbg !30
  ret i32 undef, !dbg !31
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18, !32}

!0 = !{!"0x11\0012\00clang version 3.4 (trunk 187335) (llvm/trunk 187335:187340M)\001\00\000\00\000", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/Users/nadav/file.c] [DW_LANG_C99]
!1 = !{!"file.c", !"/Users/nadav"}
!2 = !{i32 0}
!3 = !{!4}
!4 = !{!"0x2e\00depth\00depth\00\001\000\001\000\006\00256\001\001", !1, !5, !6, null, i32 (double*, i32)* @depth, null, null, !11} ; [ DW_TAG_subprogram ] [line 1] [def] [depth]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/Users/nadav/file.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{!8, !9, !8}
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = !{!"0xf\00\000\0064\0064\000\000", null, null, !10} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from double]
!10 = !{!"0x24\00double\000\0064\0064\000\000\004", null, null} ; [ DW_TAG_base_type ] [double] [line 0, size 64, align 64, offset 0, enc DW_ATE_float]
!11 = !{!12, !13, !14, !15, !16}
!12 = !{!"0x101\00A\0016777217\000", !4, !5, !9} ; [ DW_TAG_arg_variable ] [A] [line 1]
!13 = !{!"0x101\00m\0033554433\000", !4, !5, !8} ; [ DW_TAG_arg_variable ] [m] [line 1]
!14 = !{!"0x100\00y0\002\000", !4, !5, !10} ; [ DW_TAG_auto_variable ] [y0] [line 2]
!15 = !{!"0x100\00y1\002\000", !4, !5, !10} ; [ DW_TAG_auto_variable ] [y1] [line 2]
!16 = !{!"0x100\00i\003\000", !17, !5, !8} ; [ DW_TAG_auto_variable ] [i] [line 3]
!17 = !{!"0xb\003\000\000", !1, !4} ; [ DW_TAG_lexical_block ] [/Users/nadav/file.c]
!18 = !{i32 2, !"Dwarf Version", i32 2}
!19 = !MDLocation(line: 1, scope: !4)
!20 = !{double 0.000000e+00}
!21 = !MDLocation(line: 2, scope: !4)
!22 = !{double 1.000000e+00}
!23 = !MDLocation(line: 3, scope: !17)
!24 = !MDLocation(line: 4, scope: !25)
!25 = !{!"0xb\003\000\001", !1, !17} ; [ DW_TAG_lexical_block ] [/Users/nadav/file.c]
!29 = !MDLocation(line: 5, scope: !25)
!30 = !MDLocation(line: 7, scope: !4)
!31 = !MDLocation(line: 8, scope: !4)
!32 = !{i32 1, !"Debug Info Version", i32 2}
