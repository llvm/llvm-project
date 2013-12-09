; RUN: llc -split-dwarf=Enable -O0 %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t
; RUN: llvm-dwarfdump -debug-dump=all %t | FileCheck %s

; From the code:

; extern int c;
; static void foo (int p)
; {
;   int a, b; 
;   unsigned int d, e;

;   for (a = 0; a < 30; a++)
;     for (d = 0; d < 30; d++)
;       for (b = 0; b < 30; b++)
;         for (e = 0; e < 30; e++)
;           {
;             int *w = &c; 
;             *w &= p; 
;           }
; }

; void 
; bar ()
; {
;   foo (1);
; }

; compiled with:

; clang -g -S -gsplit-dwarf -O1 small.c

; CHECK: DW_AT_GNU_ranges_base

@c = external global i32

; Function Attrs: nounwind uwtable
define void @bar() #0 {
entry:
  tail call fastcc void @foo(), !dbg !27
  ret void, !dbg !28
}

; Function Attrs: nounwind uwtable
define internal fastcc void @foo() #0 {
entry:
  tail call void @llvm.dbg.value(metadata !29, i64 0, metadata !13), !dbg !30
  tail call void @llvm.dbg.value(metadata !2, i64 0, metadata !14), !dbg !31
  %c.promoted9 = load i32* @c, align 4, !dbg !32, !tbaa !33
  br label %for.cond1.preheader, !dbg !31

for.cond1.preheader:                              ; preds = %for.inc16, %entry
  %and.lcssa.lcssa.lcssa10 = phi i32 [ %c.promoted9, %entry ], [ %and, %for.inc16 ]
  %a.08 = phi i32 [ 0, %entry ], [ %inc17, %for.inc16 ]
  br label %for.cond4.preheader, !dbg !37

for.cond4.preheader:                              ; preds = %for.inc13, %for.cond1.preheader
  %and.lcssa.lcssa7 = phi i32 [ %and.lcssa.lcssa.lcssa10, %for.cond1.preheader ], [ %and, %for.inc13 ]
  %d.06 = phi i32 [ 0, %for.cond1.preheader ], [ %inc14, %for.inc13 ]
  br label %for.cond7.preheader, !dbg !38

for.cond7.preheader:                              ; preds = %for.inc10, %for.cond4.preheader
  %and.lcssa5 = phi i32 [ %and.lcssa.lcssa7, %for.cond4.preheader ], [ %and, %for.inc10 ]
  %b.03 = phi i32 [ 0, %for.cond4.preheader ], [ %inc11, %for.inc10 ]
  br label %for.body9, !dbg !39

for.body9:                                        ; preds = %for.body9, %for.cond7.preheader
  %and2 = phi i32 [ %and.lcssa5, %for.cond7.preheader ], [ %and, %for.body9 ], !dbg !40
  %e.01 = phi i32 [ 0, %for.cond7.preheader ], [ %inc, %for.body9 ]
  tail call void @llvm.dbg.value(metadata !41, i64 0, metadata !19), !dbg !40
  %and = and i32 %and2, 1, !dbg !32
  %inc = add i32 %e.01, 1, !dbg !39
  tail call void @llvm.dbg.value(metadata !{i32 %inc}, i64 0, metadata !18), !dbg !39
  %exitcond = icmp eq i32 %inc, 30, !dbg !39
  br i1 %exitcond, label %for.inc10, label %for.body9, !dbg !39

for.inc10:                                        ; preds = %for.body9
  %inc11 = add nsw i32 %b.03, 1, !dbg !38
  tail call void @llvm.dbg.value(metadata !{i32 %inc11}, i64 0, metadata !15), !dbg !38
  %exitcond11 = icmp eq i32 %inc11, 30, !dbg !38
  br i1 %exitcond11, label %for.inc13, label %for.cond7.preheader, !dbg !38

for.inc13:                                        ; preds = %for.inc10
  %inc14 = add i32 %d.06, 1, !dbg !37
  tail call void @llvm.dbg.value(metadata !{i32 %inc14}, i64 0, metadata !16), !dbg !37
  %exitcond12 = icmp eq i32 %inc14, 30, !dbg !37
  br i1 %exitcond12, label %for.inc16, label %for.cond4.preheader, !dbg !37

for.inc16:                                        ; preds = %for.inc13
  %inc17 = add nsw i32 %a.08, 1, !dbg !31
  tail call void @llvm.dbg.value(metadata !{i32 %inc17}, i64 0, metadata !14), !dbg !31
  %exitcond13 = icmp eq i32 %inc17, 30, !dbg !31
  br i1 %exitcond13, label %for.end18, label %for.cond1.preheader, !dbg !31

for.end18:                                        ; preds = %for.inc16
  store i32 %and, i32* @c, align 4, !dbg !32, !tbaa !33
  ret void, !dbg !42
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!26, !43}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.4 (trunk 191700) (llvm/trunk 191710)", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"small.dwo"} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/tmp/small.c] [DW_LANG_C99]
!1 = metadata !{metadata !"small.c", metadata !"/usr/local/google/home/echristo/tmp"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4, metadata !8}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"bar", metadata !"bar", metadata !"", i32 18, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 true, void ()* @bar, null, null, metadata !2, i32 19} ; [ DW_TAG_subprogram ] [line 18] [def] [scope 19] [bar]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/usr/local/google/home/echristo/tmp/small.c]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null}
!8 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"foo", metadata !"foo", metadata !"", i32 2, metadata !9, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 true, void ()* @foo, null, null, metadata !12, i32 3} ; [ DW_TAG_subprogram ] [line 2] [local] [def] [scope 3] [foo]
!9 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !10, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!10 = metadata !{null, metadata !11}
!11 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!12 = metadata !{metadata !13, metadata !14, metadata !15, metadata !16, metadata !18, metadata !19}
!13 = metadata !{i32 786689, metadata !8, metadata !"p", metadata !5, i32 16777218, metadata !11, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [p] [line 2]
!14 = metadata !{i32 786688, metadata !8, metadata !"a", metadata !5, i32 4, metadata !11, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [a] [line 4]
!15 = metadata !{i32 786688, metadata !8, metadata !"b", metadata !5, i32 4, metadata !11, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [b] [line 4]
!16 = metadata !{i32 786688, metadata !8, metadata !"d", metadata !5, i32 5, metadata !17, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [d] [line 5]
!17 = metadata !{i32 786468, null, null, metadata !"unsigned int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ] [unsigned int] [line 0, size 32, align 32, offset 0, enc DW_ATE_unsigned]
!18 = metadata !{i32 786688, metadata !8, metadata !"e", metadata !5, i32 5, metadata !17, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [e] [line 5]
!19 = metadata !{i32 786688, metadata !20, metadata !"w", metadata !5, i32 12, metadata !25, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [w] [line 12]
!20 = metadata !{i32 786443, metadata !1, metadata !21, i32 11, i32 0, i32 4} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/small.c]
!21 = metadata !{i32 786443, metadata !1, metadata !22, i32 10, i32 0, i32 3} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/small.c]
!22 = metadata !{i32 786443, metadata !1, metadata !23, i32 9, i32 0, i32 2} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/small.c]
!23 = metadata !{i32 786443, metadata !1, metadata !24, i32 8, i32 0, i32 1} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/small.c]
!24 = metadata !{i32 786443, metadata !1, metadata !8, i32 7, i32 0, i32 0} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/small.c]
!25 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !11} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!26 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!27 = metadata !{i32 20, i32 0, metadata !4, null}
!28 = metadata !{i32 21, i32 0, metadata !4, null}
!29 = metadata !{i32 1}
!30 = metadata !{i32 2, i32 0, metadata !8, null}
!31 = metadata !{i32 7, i32 0, metadata !24, null}
!32 = metadata !{i32 13, i32 0, metadata !20, null}
!33 = metadata !{metadata !34, metadata !34, i64 0}
!34 = metadata !{metadata !"int", metadata !35, i64 0}
!35 = metadata !{metadata !"omnipotent char", metadata !36, i64 0}
!36 = metadata !{metadata !"Simple C/C++ TBAA"}
!37 = metadata !{i32 8, i32 0, metadata !23, null} ; [ DW_TAG_imported_declaration ]
!38 = metadata !{i32 9, i32 0, metadata !22, null}
!39 = metadata !{i32 10, i32 0, metadata !21, null}
!40 = metadata !{i32 12, i32 0, metadata !20, null}
!41 = metadata !{i32* @c}
!42 = metadata !{i32 15, i32 0, metadata !8, null}
!43 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
