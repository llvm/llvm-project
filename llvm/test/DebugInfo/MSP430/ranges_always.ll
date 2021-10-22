; RUN: llc -O0 %s -mtriple=msp430 -filetype=obj -o - -minimize-addr-in-v5=Ranges \
; RUN:   | llvm-dwarfdump -debug-info -debug-addr -debug-rnglists -v - \
; RUN:   | FileCheck --check-prefix=CHECK --check-prefix=RNG \
; RUN:     --implicit-check-not=DW_TAG --implicit-check-not=NULL --implicit-check-not=_pc %s

; RUN: llc -O0 %s -mtriple=msp430 -filetype=obj -o - -minimize-addr-in-v5=Expressions \
; RUN:   | llvm-dwarfdump -debug-info -debug-addr -debug-rnglists -v - \
; RUN:   | FileCheck --check-prefix=CHECK --check-prefix=EXPRORFORM --check-prefix=EXPR\
; RUN:     --implicit-check-not=DW_TAG --implicit-check-not=NULL --implicit-check-not=_pc %s

; RUN: llc -O0 %s -mtriple=msp430 -filetype=obj -o - -minimize-addr-in-v5=Form \
; RUN:   | llvm-dwarfdump -debug-info -debug-addr -debug-rnglists -v - \
; RUN:   | FileCheck --check-prefix=CHECK --check-prefix=EXPRORFORM --check-prefix=FORM \
; RUN:     --implicit-check-not=DW_TAG --implicit-check-not=NULL --implicit-check-not=_pc %s

; Ported from X86 test to cover 2-byte address size case

; Generated from the following source. f4 is used to put a hole in the CU
; ranges while keeping f2 and f4 in the same section (as opposed to
; -ffunction-sections, which would produce CU ranges, but each function would
; be in a different section, so unable to share addresses). The call to f1 at
; the start of f3 ensures the range for the inlined subroutine doesn't share
; the starting address with f3 (so it can be improved by using a rnglist to
; allow it to share an address it wouldn't already be sharing).

; Without f6 being in another section, technically we could use a non-zero CU
; low_pc that could act as a base address for all the addresses in the CU & avoid
; the need for these forced rnglists - we don't do that currently, but f6 ensures
; that this test will remain meaningful even if that improvement is made in the
; future. (implementing that would require detecting that all the addresses in
; the CU ranges are in the same section, then picking the lowest such address as
; the base address to make all other addresses relative to)

; IR from the following, compiled with:
; $ clang -g -c -gdwarf-5 -O1
; __attribute__((optnone)) void f1() { }
; __attribute__((always_inline)) inline void f2() {
;   f1();
; }
; void f3() {
;   f1();
;   f2();
; }
; __attribute__((nodebug)) void f4() {
; }
; void f5() {
; }
; __attribute__((section(".other"))) void f6() {
; }

; CHECK-LABEL: .debug_info contents:
; CHECK: DW_TAG_compile_unit
; CHECK:   DW_AT_low_pc
; CHECK-SAME: (0x0000)
; RNG:     DW_AT_ranges
; RNG-SAME:  (indexed (0x3) rangelist = [[CU_RANGE:.*]]
; EXPRORFORM: DW_AT_ranges
; EXPRORFORM-SAME: (indexed (0x0) rangelist = [[CU_RANGE:.*]]
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_low_pc
; CHECK-SAME:  (indexed (00000000) address = 0x0000 ".text")
; CHECK:     DW_AT_high_pc
; CHECK-SAME:  (0x00000002)
; CHECK:     DW_AT_name
; CHECK-SAME:  "f1"
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_name
; CHECK-SAME:  "f2"
; CHECK:   DW_TAG_subprogram
; EXPR:      DW_AT_low_pc
; EXPR-SAME:   (DW_OP_addrx 0x0, DW_OP_const4u 0x2, DW_OP_plus)
; FORM:      DW_AT_low_pc
; FORM-SAME:   [DW_FORM_LLVM_addrx_offset] (indexed (00000000) + 0x2 address = 0x0002 ".text")
; EXPRORFORM: DW_AT_high_pc
; EXPRORFORM-SAME: (0x0000000a)
; RNG:       DW_AT_ranges
; RNG-SAME:    (indexed (0x0) rangelist = [[F3_RANGE:.*]]
; CHECK:     DW_AT_name
; CHECK-SAME: "f3"
; CHECK:     DW_TAG_inlined_subroutine
; EXPR:        DW_AT_low_pc
; EXPR-SAME:     [DW_FORM_exprloc] (DW_OP_addrx 0x0, DW_OP_const4u 0x6, DW_OP_plus)
; FORM:        DW_AT_low_pc
; FORM-SAME:     [DW_FORM_LLVM_addrx_offset] (indexed (00000000) + 0x6 address = 0x0006 ".text")
; EXPRORFORM:  DW_AT_high_pc
; EXPRORFORM-SAME: (0x00000004)
; RNG:         DW_AT_ranges
; RNG-SAME:      (indexed (0x1) rangelist = [[INL_RANGE:.*]]
; CHECK:     DW_TAG_call_site
; RNG:         DW_AT_call_return_pc
; RNG-SAME:      (indexed (00000001) address = 0x0006 ".text")
; EXPR:        DW_AT_call_return_pc
; EXPR-SAME:     [DW_FORM_exprloc] (DW_OP_addrx 0x0, DW_OP_const4u 0x6, DW_OP_plus)
; FORM:        DW_AT_call_return_pc
; FORM-SAME:     [DW_FORM_LLVM_addrx_offset] (indexed (00000000) + 0x6 address = 0x0006 ".text")
; CHECK:     DW_TAG_call_site
; RNG:         DW_AT_call_return_pc
; RNG-SAME:      (indexed (00000002) address = 0x000a ".text")
; EXPR:        DW_AT_call_return_pc
; EXPR-SAME:     [DW_FORM_exprloc] (DW_OP_addrx 0x0, DW_OP_const4u 0xa, DW_OP_plus)
; FORM:        DW_AT_call_return_pc
; FORM-SAME:     [DW_FORM_LLVM_addrx_offset] (indexed (00000000) + 0xa address = 0x000a ".text")
; CHECK:     NULL
; CHECK:   DW_TAG_subprogram
; EXPR:      DW_AT_low_pc
; EXPR-SAME:   [DW_FORM_exprloc] (DW_OP_addrx 0x0, DW_OP_const4u 0xe, DW_OP_plus)
; FORM:      DW_AT_low_pc
; FORM-SAME:   [DW_FORM_LLVM_addrx_offset] (indexed (00000000) + 0xe address = 0x000e ".text")
; EXPRORFORM: DW_AT_high_pc
; EXPRORFORM-SAME: (0x00000002)
; RNG:       DW_AT_ranges
; RNG-SAME:    (indexed (0x2) rangelist = [[F5_RANGE:.*]]
; CHECK:     DW_AT_name
; CHECK-SAME:  "f5"
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_low_pc [DW_FORM_addrx]    (indexed (
; RNG-SAME: 00000003
; EXPRORFORM-SAME: 00000001
; CHECK: ) address = 0x0000 ".other")
; CHECK:     DW_AT_high_pc
; CHECK-SAME:  (0x00000006)
; CHECK:     DW_AT_name
; CHECK-SAME: "f6"
; CHECK:       DW_TAG_inlined_subroutine
; RNG:           DW_AT_low_pc
; RNG-SAME:        (indexed (00000003) address = 0x0000 ".other")
; EXPRORFORM:    DW_AT_low_pc
; EXPRORFORM-SAME: (indexed (00000001) address = 0x0000 ".other")
; CHECK:         DW_AT_high_pc
; CHECK-SAME:      (0x00000004)
; CHECK:       DW_TAG_call_site
; CHECK:         DW_AT_call_return_pc
; CHECK:       NULL
; CHECK:   NULL

; CHECK-LABEL: .debug_addr contents:
; CHECK: 0x00000000: Address table
; CHECK-NEXT: Addrs: [
; CHECK-NEXT: 0x0000
; RNG-NEXT:   0x0006
; RNG-NEXT:   0x000a
; CHECK-NEXT: 0x0000
; RNG-NEXT:   0x0004
; CHECK-NEXT: ]

; CHECK-LABEL: .debug_rnglists contents:
; RNG: 0x00000000: range list header: {{.*}}, offset_entry_count = 0x00000004
; EXPRORFORM: 0x00000000: range list header: {{.*}}, offset_entry_count = 0x00000001
; CHECK: ranges:
; RNG-NEXT:   [[F3_RANGE]]: [DW_RLE_base_addressx]:
; RNG-SAME:                   0x0000
; RNG-NEXT:                 [DW_RLE_offset_pair  ]
; RNG-NEXT:                 [DW_RLE_end_of_list  ]

; RNG-NEXT:   [[INL_RANGE]]: [DW_RLE_base_addressx]:
; RNG-SAME:                    0x0000
; RNG-NEXT:                  [DW_RLE_offset_pair  ]
; RNG-NEXT:                  [DW_RLE_end_of_list  ]

; RNG-NEXT:   [[F5_RANGE]]: [DW_RLE_base_addressx]:
; RNG-SAME:                   0x0000
; RNG-NEXT:                 [DW_RLE_offset_pair  ]
; RNG-NEXT:                 [DW_RLE_end_of_list  ]

; CHECK-NEXT: [[CU_RANGE]]: [DW_RLE_base_addressx]:
; CHECK-SAME:                 0x0000
; CHECK-NEXT:               [DW_RLE_offset_pair  ]
; CHECK-NEXT:               [DW_RLE_offset_pair  ]
; RNG-NEXT:                 [DW_RLE_startx_length]:
; RNG-SAME:                   0x0003
; EXPRORFORM-NEXT:          [DW_RLE_startx_length]:
; EXPRORFORM-SAME:            0x0001
; CHECK-NEXT:               [DW_RLE_end_of_list  ]

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @_Z2f1v() local_unnamed_addr #0 !dbg !7 {
entry:
  ret void, !dbg !12
}

; Function Attrs: mustprogress nounwind uwtable
define dso_local void @_Z2f3v() local_unnamed_addr #1 !dbg !13 {
entry:
  call void @_Z2f1v(), !dbg !14
  call void @_Z2f1v() #3, !dbg !15
  ret void, !dbg !18
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone uwtable willreturn
define dso_local void @_Z2f4v() local_unnamed_addr #2 {
entry:
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone uwtable willreturn
define dso_local void @_Z2f5v() local_unnamed_addr #2 !dbg !19 {
entry:
  ret void, !dbg !20
}

; Function Attrs: mustprogress nounwind uwtable
define dso_local void @_Z2f6v() local_unnamed_addr #1 section ".other" !dbg !21 {
entry:
  call void @_Z2f1v() #3, !dbg !22
  ret void, !dbg !24
}

attributes #0 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { mustprogress nofree norecurse nosync nounwind readnone uwtable willreturn "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 14.0.0 (git@github.com:llvm/llvm-project.git e2c3dc6fc76e767f08249f6d2c36e41660a4e331)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "/usr/local/google/home/blaikie/dev/scratch/test.cpp", directory: "/usr/local/google/home/blaikie/dev/llvm/src", checksumkind: CSK_MD5, checksum: "e70db21a276125757057e729999c09c7")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 1}
!6 = !{!"clang version 14.0.0 (git@github.com:llvm/llvm-project.git e2c3dc6fc76e767f08249f6d2c36e41660a4e331)"}
!7 = distinct !DISubprogram(name: "f1", linkageName: "_Z2f1v", scope: !8, file: !8, line: 1, type: !9, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DIFile(filename: "scratch/test.cpp", directory: "/usr/local/google/home/blaikie/dev", checksumkind: CSK_MD5, checksum: "e70db21a276125757057e729999c09c7")
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !{}
!12 = !DILocation(line: 1, column: 38, scope: !7)
!13 = distinct !DISubprogram(name: "f3", linkageName: "_Z2f3v", scope: !8, file: !8, line: 5, type: !9, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!14 = !DILocation(line: 6, column: 3, scope: !13)
!15 = !DILocation(line: 3, column: 3, scope: !16, inlinedAt: !17)
!16 = distinct !DISubprogram(name: "f2", linkageName: "_Z2f2v", scope: !8, file: !8, line: 2, type: !9, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!17 = distinct !DILocation(line: 7, column: 3, scope: !13)
!18 = !DILocation(line: 8, column: 1, scope: !13)
!19 = distinct !DISubprogram(name: "f5", linkageName: "_Z2f5v", scope: !8, file: !8, line: 11, type: !9, scopeLine: 11, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!20 = !DILocation(line: 12, column: 1, scope: !19)
!21 = distinct !DISubprogram(name: "f6", linkageName: "_Z2f6v", scope: !8, file: !8, line: 13, type: !9, scopeLine: 13, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!22 = !DILocation(line: 3, column: 3, scope: !16, inlinedAt: !23)
!23 = distinct !DILocation(line: 14, column: 3, scope: !21)
!24 = !DILocation(line: 15, column: 1, scope: !21)
