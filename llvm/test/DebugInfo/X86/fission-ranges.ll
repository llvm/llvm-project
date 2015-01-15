; RUN: llc -split-dwarf=Enable -O0 %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t
; RUN: llvm-dwarfdump %t | FileCheck %s
; RUN: llvm-objdump -h %t | FileCheck --check-prefix=HDR %s

; CHECK: .debug_info contents:
; CHECK: DW_TAG_compile_unit
; CHECK-NEXT: DW_AT_stmt_list
; CHECK-NEXT: DW_AT_GNU_dwo_name
; CHECK-NEXT: DW_AT_comp_dir
; CHECK-NEXT: DW_AT_GNU_dwo_id
; CHECK-NEXT: DW_AT_GNU_addr_base [DW_FORM_sec_offset]                   (0x00000000)


; CHECK: .debug_info.dwo contents:
; CHECK: DW_AT_location [DW_FORM_sec_offset]   ([[A:0x[0-9a-z]*]])
; CHECK: DW_AT_location [DW_FORM_sec_offset]   ([[E:0x[0-9a-z]*]])
; CHECK: DW_AT_location [DW_FORM_sec_offset]   ([[B:0x[0-9a-z]*]])
; CHECK: DW_AT_location [DW_FORM_sec_offset]   ([[D:0x[0-9a-z]*]])
; CHECK: DW_AT_ranges [DW_FORM_sec_offset]   (0x00000000
; CHECK: .debug_loc contents:
; CHECK-NOT: Beginning address offset
; CHECK: .debug_loc.dwo contents:

; Don't assume these locations are entirely correct - feel free to update them
; if they've changed due to a bugfix, change in register allocation, etc.

; CHECK: [[A]]: Beginning address index: 2
; CHECK-NEXT:                    Length: 190
; CHECK-NEXT:      Location description: 11 00
; CHECK-NEXT: {{^$}}
; CHECK-NEXT:   Beginning address index: 3
; CHECK-NEXT:                    Length: 23
; CHECK-NEXT:      Location description: 50 93 04
; CHECK: [[E]]: Beginning address index: 4
; CHECK-NEXT:                    Length: 21
; CHECK-NEXT:      Location description: 50 93 04
; CHECK: [[B]]: Beginning address index: 5
; CHECK-NEXT:                    Length: 19
; CHECK-NEXT:      Location description: 50 93 04
; CHECK: [[D]]: Beginning address index: 6
; CHECK-NEXT:                    Length: 23
; CHECK-NEXT:      Location description: 50 93 04

; Make sure we don't produce any relocations in any .dwo section (though in particular, debug_info.dwo)
; HDR-NOT: .rela.{{.*}}.dwo

; Make sure we have enough stuff in the debug_addr to cover the address indexes
; (6 is the last index in debug_loc.dwo, making 7 entries of 8 bytes each, 7 * 8
; == 56 base 10 == 38 base 16)

; HDR: .debug_addr 00000038
; HDR-NOT: .rela.{{.*}}.dwo

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
  tail call void @llvm.dbg.value(metadata i32 1, i64 0, metadata !13, metadata !{!"0x102"}), !dbg !30
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !14, metadata !{!"0x102"}), !dbg !31
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
  tail call void @llvm.dbg.value(metadata i32* @c, i64 0, metadata !19, metadata !{!"0x102"}), !dbg !40
  %and = and i32 %and2, 1, !dbg !32
  %inc = add i32 %e.01, 1, !dbg !39
  tail call void @llvm.dbg.value(metadata i32 %inc, i64 0, metadata !18, metadata !{!"0x102"}), !dbg !39
  %exitcond = icmp eq i32 %inc, 30, !dbg !39
  br i1 %exitcond, label %for.inc10, label %for.body9, !dbg !39

for.inc10:                                        ; preds = %for.body9
  %inc11 = add nsw i32 %b.03, 1, !dbg !38
  tail call void @llvm.dbg.value(metadata i32 %inc11, i64 0, metadata !15, metadata !{!"0x102"}), !dbg !38
  %exitcond11 = icmp eq i32 %inc11, 30, !dbg !38
  br i1 %exitcond11, label %for.inc13, label %for.cond7.preheader, !dbg !38

for.inc13:                                        ; preds = %for.inc10
  %inc14 = add i32 %d.06, 1, !dbg !37
  tail call void @llvm.dbg.value(metadata i32 %inc14, i64 0, metadata !16, metadata !{!"0x102"}), !dbg !37
  %exitcond12 = icmp eq i32 %inc14, 30, !dbg !37
  br i1 %exitcond12, label %for.inc16, label %for.cond4.preheader, !dbg !37

for.inc16:                                        ; preds = %for.inc13
  %inc17 = add nsw i32 %a.08, 1, !dbg !31
  tail call void @llvm.dbg.value(metadata i32 %inc17, i64 0, metadata !14, metadata !{!"0x102"}), !dbg !31
  %exitcond13 = icmp eq i32 %inc17, 30, !dbg !31
  br i1 %exitcond13, label %for.end18, label %for.cond1.preheader, !dbg !31

for.end18:                                        ; preds = %for.inc16
  store i32 %and, i32* @c, align 4, !dbg !32, !tbaa !33
  ret void, !dbg !42
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!26, !43}

!0 = !{!"0x11\0012\00clang version 3.4 (trunk 191700) (llvm/trunk 191710)\001\00\000\00small.dwo\000", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/tmp/small.c] [DW_LANG_C99]
!1 = !{!"small.c", !"/usr/local/google/home/echristo/tmp"}
!2 = !{}
!3 = !{!4, !8}
!4 = !{!"0x2e\00bar\00bar\00\0018\000\001\000\006\000\001\0019", !1, !5, !6, null, void ()* @bar, null, null, !2} ; [ DW_TAG_subprogram ] [line 18] [def] [scope 19] [bar]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/usr/local/google/home/echristo/tmp/small.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null}
!8 = !{!"0x2e\00foo\00foo\00\002\001\001\000\006\00256\001\003", !1, !5, !9, null, void ()* @foo, null, null, !12} ; [ DW_TAG_subprogram ] [line 2] [local] [def] [scope 3] [foo]
!9 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !10, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!10 = !{null, !11}
!11 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!12 = !{!13, !14, !15, !16, !18, !19}
!13 = !{!"0x101\00p\0016777218\000", !8, !5, !11} ; [ DW_TAG_arg_variable ] [p] [line 2]
!14 = !{!"0x100\00a\004\000", !8, !5, !11} ; [ DW_TAG_auto_variable ] [a] [line 4]
!15 = !{!"0x100\00b\004\000", !8, !5, !11} ; [ DW_TAG_auto_variable ] [b] [line 4]
!16 = !{!"0x100\00d\005\000", !8, !5, !17} ; [ DW_TAG_auto_variable ] [d] [line 5]
!17 = !{!"0x24\00unsigned int\000\0032\0032\000\000\007", null, null} ; [ DW_TAG_base_type ] [unsigned int] [line 0, size 32, align 32, offset 0, enc DW_ATE_unsigned]
!18 = !{!"0x100\00e\005\000", !8, !5, !17} ; [ DW_TAG_auto_variable ] [e] [line 5]
!19 = !{!"0x100\00w\0012\000", !20, !5, !25} ; [ DW_TAG_auto_variable ] [w] [line 12]
!20 = !{!"0xb\0011\000\004", !1, !21} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/small.c]
!21 = !{!"0xb\0010\000\003", !1, !22} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/small.c]
!22 = !{!"0xb\009\000\002", !1, !23} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/small.c]
!23 = !{!"0xb\008\000\001", !1, !24} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/small.c]
!24 = !{!"0xb\007\000\000", !1, !8} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/small.c]
!25 = !{!"0xf\00\000\0064\0064\000\000", null, null, !11} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!26 = !{i32 2, !"Dwarf Version", i32 4}
!27 = !MDLocation(line: 20, scope: !4)
!28 = !MDLocation(line: 21, scope: !4)
!29 = !{i32 1}
!30 = !MDLocation(line: 2, scope: !8)
!31 = !MDLocation(line: 7, scope: !24)
!32 = !MDLocation(line: 13, scope: !20)
!33 = !{!34, !34, i64 0}
!34 = !{!"int", !35, i64 0}
!35 = !{!"omnipotent char", !36, i64 0}
!36 = !{!"Simple C/C++ TBAA"}
!37 = !MDLocation(line: 8, scope: !23)
!38 = !MDLocation(line: 9, scope: !22)
!39 = !MDLocation(line: 10, scope: !21)
!40 = !MDLocation(line: 12, scope: !20)
!41 = !{i32* @c}
!42 = !MDLocation(line: 15, scope: !8)
!43 = !{i32 1, !"Debug Info Version", i32 2}
!44 = !{i32 0}
