; RUN: llc -O0 %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o - -minimize-addr-in-v5=Ranges \
; RUN:   | llvm-dwarfdump -debug-info -debug-addr -debug-rnglists -v - \
; RUN:   | FileCheck --check-prefix=CHECK --check-prefix=RNG \
; RUN:     --implicit-check-not=DW_TAG --implicit-check-not=NULL --implicit-check-not=_pc %s

; RUN: llc -O0 %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o - -minimize-addr-in-v5=Expressions \
; RUN:   | llvm-dwarfdump -debug-info -debug-addr -debug-rnglists -v - \
; RUN:   | FileCheck --check-prefix=CHECK --check-prefix=EXPRORFORM --check-prefix=EXPR\
; RUN:     --implicit-check-not=DW_TAG --implicit-check-not=NULL --implicit-check-not=_pc %s

; RUN: llc -O0 %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o - -minimize-addr-in-v5=Form \
; RUN:   | llvm-dwarfdump -debug-info -debug-addr -debug-rnglists -v - \
; RUN:   | FileCheck --check-prefix=CHECK --check-prefix=EXPRORFORM --check-prefix=FORM \
; RUN:     --implicit-check-not=DW_TAG --implicit-check-not=NULL --implicit-check-not=_pc %s


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

; void f1();
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
; CHECK:   DW_AT_low_pc [DW_FORM_addr] (0x0000000000000000)
; RNG:     DW_AT_ranges [DW_FORM_rnglistx]   (indexed (0x2) rangelist = [[CU_RANGE:.*]]
; EXPRORFORM: DW_AT_ranges [DW_FORM_rnglistx]   (indexed (0x0) rangelist = [[CU_RANGE:.*]]
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}} "f2"
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_low_pc [DW_FORM_addrx]    (indexed (00000000) address = 0x0000000000000000 ".text")
; CHECK:     DW_AT_high_pc [DW_FORM_data4]   (0x00000010)
; CHECK:     DW_TAG_inlined_subroutine
; EXPR:        DW_AT_low_pc [DW_FORM_exprloc] (DW_OP_addrx 0x0, DW_OP_const4u 0x9, DW_OP_plus)
; FORM:        DW_AT_low_pc [DW_FORM_LLVM_addrx_offset] (indexed (00000000) + 0x9 address = 0x0000000000000009 ".text")
; EXPRORFORM:  DW_AT_high_pc [DW_FORM_data4]   (0x00000005)
; RNG:         DW_AT_ranges [DW_FORM_rnglistx] (indexed (0x0) rangelist = [[INL_RANGE:.*]]
; CHECK:     DW_TAG_call_site
; RNG:         DW_AT_call_return_pc [DW_FORM_addrx]  (indexed (00000001) address = 0x0000000000000009 ".text")
; EXPR:        DW_AT_call_return_pc [DW_FORM_exprloc] (DW_OP_addrx 0x0, DW_OP_const4u 0x9, DW_OP_plus)
; FORM:        DW_AT_call_return_pc [DW_FORM_LLVM_addrx_offset] (indexed (00000000) + 0x9 address = 0x0000000000000009 ".text")
; CHECK:     DW_TAG_call_site
; RNG:         DW_AT_call_return_pc [DW_FORM_addrx]  (indexed (00000002) address = 0x000000000000000e ".text")
; EXPR:        DW_AT_call_return_pc [DW_FORM_exprloc] (DW_OP_addrx 0x0, DW_OP_const4u 0xe, DW_OP_plus)
; FORM:        DW_AT_call_return_pc [DW_FORM_LLVM_addrx_offset] (indexed (00000000) + 0xe address = 0x000000000000000e ".text")
; CHECK:     NULL
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}} "f1"
; CHECK:   DW_TAG_subprogram
; EXPR:      DW_AT_low_pc [DW_FORM_exprloc] (DW_OP_addrx 0x0, DW_OP_const4u 0x20, DW_OP_plus)
; FORM:      DW_AT_low_pc [DW_FORM_LLVM_addrx_offset] (indexed (00000000) + 0x20 address = 0x0000000000000020 ".text")
; EXPRORFORM: DW_AT_high_pc [DW_FORM_data4]   (0x00000006)
; RNG:       DW_AT_ranges [DW_FORM_rnglistx] (indexed (0x1) rangelist = [[F5_RANGE:.*]]
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_low_pc [DW_FORM_addrx]    (indexed (
; RNG-SAME: 00000003
; EXPRORFORM-SAME: 00000001
; CHECK: ) address = 0x0000000000000000 ".other")
; CHECK:     DW_AT_high_pc [DW_FORM_data4]   (0x00000006)
; CHECK:   NULL

; CHECK-LABEL: .debug_addr contents:
; CHECK: 0x00000000: Address table
; CHECK-NEXT: Addrs: [
; CHECK-NEXT: 0x0000000000000000
; RNG-NEXT:   0x0000000000000009
; RNG-NEXT:   0x000000000000000e
; CHECK-NEXT: 0x0000000000000000
; CHECK-NEXT: ]

; CHECK-LABEL: .debug_rnglists contents:
; RNG: 0x00000000: range list header: {{.*}}, offset_entry_count = 0x00000003
; EXPRORFORM: 0x00000000: range list header: {{.*}}, offset_entry_count = 0x00000001
; CHECK: ranges:
; RNG-NEXT:   [[INL_RANGE]]: [DW_RLE_base_addressx]:  0x0000000000000000
; RNG-NEXT:                  [DW_RLE_offset_pair  ]
; RNG-NEXT:                  [DW_RLE_end_of_list  ]

; RNG-NEXT:   [[F5_RANGE]]: [DW_RLE_base_addressx]:  0x0000000000000000
; RNG-NEXT:                 [DW_RLE_offset_pair  ]
; RNG-NEXT:                 [DW_RLE_end_of_list  ]

; CHECK-NEXT: [[CU_RANGE]]: [DW_RLE_base_addressx]:  0x0000000000000000
; CHECK-NEXT:               [DW_RLE_offset_pair  ]
; CHECK-NEXT:               [DW_RLE_offset_pair  ]
; RNG-NEXT:                 [DW_RLE_startx_length]:  0x0000000000000003
; EXPRORFORM-NEXT:          [DW_RLE_startx_length]:  0x0000000000000001
; CHECK-NEXT:               [DW_RLE_end_of_list  ]

; Function Attrs: noinline optnone uwtable mustprogress
define dso_local void @_Z2f3v() #0 !dbg !7 {
entry:
  call void @_Z2f1v(), !dbg !10
  call void @_Z2f1v(), !dbg !11
  ret void, !dbg !14
}

declare !dbg !19 dso_local void @_Z2f1v() #1

; Function Attrs: noinline nounwind optnone uwtable mustprogress
define dso_local void @_Z2f4v() #2 {
entry:
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable mustprogress
define dso_local void @_Z2f5v() #2 !dbg !15 {
entry:
  ret void, !dbg !16
}

; Function Attrs: noinline nounwind optnone uwtable mustprogress
define dso_local void @_Z2f6v() #2 section ".other" !dbg !17 {
entry:
  ret void, !dbg !18
}

attributes #0 = { noinline optnone uwtable mustprogress "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { noinline nounwind optnone uwtable mustprogress "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 12.0.0 (git@github.com:llvm/llvm-project.git 79afdd7d36b814942ec7f2f577d0443f6aecc939)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "ranges_always.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch", checksumkind: CSK_MD5, checksum: "25fb47763043609a0aac0ab69baa803d")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 12.0.0 (git@github.com:llvm/llvm-project.git 79afdd7d36b814942ec7f2f577d0443f6aecc939)"}
!7 = distinct !DISubprogram(name: "f3", linkageName: "_Z2f3v", scope: !1, file: !1, line: 5, type: !8, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocation(line: 6, column: 3, scope: !7)
!11 = !DILocation(line: 3, column: 3, scope: !12, inlinedAt: !13)
!12 = distinct !DISubprogram(name: "f2", linkageName: "_Z2f2v", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!13 = distinct !DILocation(line: 7, column: 3, scope: !7)
!14 = !DILocation(line: 8, column: 1, scope: !7)
!15 = distinct !DISubprogram(name: "f5", linkageName: "_Z2f5v", scope: !1, file: !1, line: 11, type: !8, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!16 = !DILocation(line: 12, column: 1, scope: !15)
!17 = distinct !DISubprogram(name: "f6", linkageName: "_Z2f6v", scope: !1, file: !1, line: 13, type: !8, scopeLine: 13, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!18 = !DILocation(line: 14, column: 1, scope: !17)
!19 = !DISubprogram(name: "f1", linkageName: "_Z2f1v", scope: !1, file: !1, line: 1, type: !8, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
