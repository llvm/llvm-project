; RUN: llc -filetype=asm -mtriple=x86_64-pc-linux-gnu %s -o - | FileCheck %s
; RUN: llc -filetype=asm -mtriple=x86_64-pc-linux-gnu %s -o - -dwarf-version 5 | FileCheck --check-prefix=DWARF5 %s

; Group ranges in a range list that apply to the same section and use a base
; address selection entry to reduce the number of relocations to one reloc per
; section per range list. DWARF5 debug_rnglist (with *x variants) are more
; efficient than this in terms of relocations, but it's still better than one
; reloc per entry in a range list.

; This is an object/executable size tradeoff - shrinking objects, but growing
; the linked executable. In one large binary tested, total object size (not just
; debug info) shrank by 16%, entirely relocation entries. Linked executable
; grew by 4%. This was with compressed debug info in the objects, uncompressed
; in the linked executable. Without compression in the objects, the win would be
; smaller (the growth of debug_ranges itself would be more significant).

; This is a merged module containing two CUs, one that uses range base address
; specifiers and exercises different cases there, and another that does not

; ranges.cpp
; Single range entry
; __attribute__((section("a"))) void f1() {}
; Single address with two ranges due to the whole caused by f3
; __attribute__((section("b"))) void f2() {}
; __attribute__((section("b"))) __attribute__((nodebug)) void f3() {}
; __attribute__((section("b"))) void f4() {}
; Reset the base address & emit a couple more single range entries
; __attribute__((section("c"))) void f5() {}
; __attribute__((section("d"))) void f6() {}
; ranges_no_base.cpp:
; Include enough complexity to cause ranges to be emitted, so it can be checked
; that those ranges don't use base address specifiers
; __attribute__((section("e"))) void f7() {}
; __attribute__((section("f"))) void f8() {}

; CHECK: {{^.Ldebug_ranges0}}
; CHECK-NEXT:   .quad   -1
; CHECK-NEXT:   .quad   .Lfunc_begin0
; CHECK-NEXT:   .quad   .Lfunc_begin0-.Lfunc_begin0
; CHECK-NEXT:   .quad   .Lfunc_end0-.Lfunc_begin0
; CHECK-NEXT:   .quad   -1
; CHECK-NEXT:   .quad   .Lfunc_begin1
; CHECK-NEXT:   .quad   .Lfunc_begin1-.Lfunc_begin1
; CHECK-NEXT:   .quad   .Lfunc_end1-.Lfunc_begin1
; CHECK-NEXT:   .quad   .Lfunc_begin3-.Lfunc_begin1
; CHECK-NEXT:   .quad   .Lfunc_end3-.Lfunc_begin1
; CHECK-NEXT:   .quad   -1
; CHECK-NEXT:   .quad   .Lfunc_begin4
; CHECK-NEXT:   .quad   .Lfunc_begin4-.Lfunc_begin4
; CHECK-NEXT:   .quad   .Lfunc_end4-.Lfunc_begin4
; CHECK-NEXT:   .quad   -1
; CHECK-NEXT:   .quad   .Lfunc_begin5
; CHECK-NEXT:   .quad   .Lfunc_begin5-.Lfunc_begin5
; CHECK-NEXT:   .quad   .Lfunc_end5-.Lfunc_begin5
; CHECK-NEXT:   .quad   0
; CHECK-NEXT:   .quad   0
; CHECK-NEXT: {{^.Ldebug_ranges1}}
; CHECK-NEXT:   .quad   .Lfunc_begin6
; CHECK-NEXT:   .quad   .Lfunc_end6
; CHECK-NEXT:   .quad   .Lfunc_begin7
; CHECK-NEXT:   .quad   .Lfunc_end7

; DWARF5: {{^.Ldebug_ranges0}}
; DWARF5-NEXT:                                      # DW_RLE_startx_length
; DWARF5-NEXT: .byte 0                              #   start index
; DWARF5-NEXT: .uleb128 .Lfunc_end0-.Lfunc_begin0   #   length
; DWARF5-NEXT:                                      # DW_RLE_base_addressx
; DWARF5-NEXT: .byte 1                              #   base address index
; DWARF5-NEXT:                                      # DW_RLE_offset_pair
; DWARF5-NEXT: .uleb128 .Lfunc_begin1-.Lfunc_begin1 #   starting offset
; DWARF5-NEXT: .uleb128 .Lfunc_end1-.Lfunc_begin1   #   ending offset
; DWARF5-NEXT:                                      # DW_RLE_offset_pair
; DWARF5-NEXT: .uleb128 .Lfunc_begin3-.Lfunc_begin1 #   starting offset
; DWARF5-NEXT: .uleb128 .Lfunc_end3-.Lfunc_begin1   #   ending offset
; DWARF5-NEXT:                                      # DW_RLE_startx_length
; DWARF5-NEXT: .byte 3                              #   start index
; DWARF5-NEXT: .uleb128 .Lfunc_end4-.Lfunc_begin4   #   length
; DWARF5-NEXT:                                      # DW_RLE_startx_length
; DWARF5-NEXT: .byte 4                              #   start index
; DWARF5-NEXT: .uleb128 .Lfunc_end5-.Lfunc_begin5   #   length
; DWARF5-NEXT:                                      # DW_RLE_end_of_list
; DWARF5-NEXT: {{^.Ldebug_ranges1}}
; DWARF5-NEXT:                                      # DW_RLE_startx_length
; DWARF5-NEXT: .byte 5                              #   start index
; DWARF5-NEXT: .uleb128 .Lfunc_end6-.Lfunc_begin6   #   length
; DWARF5-NEXT:                                      # DW_RLE_startx_length
; DWARF5-NEXT: .byte 6                              #   start index
; DWARF5-NEXT: .uleb128 .Lfunc_end7-.Lfunc_begin7   #   length
; DWARF5-NEXT:                                      # DW_RLE_end_of_list

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @_Z2f1v() section "a" !dbg !9 {
entry:
  ret void, !dbg !12
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @_Z2f2v() section "b" !dbg !13 {
entry:
  ret void, !dbg !14
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @_Z2f3v() section "b" {
entry:
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @_Z2f4v() section "b" !dbg !15 {
entry:
  ret void, !dbg !16
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @_Z2f5v() section "c" !dbg !17 {
entry:
  ret void, !dbg !18
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @_Z2f6v() section "d" !dbg !19 {
entry:
  ret void, !dbg !20
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @_Z2f7v() section "e" !dbg !21 {
entry:
  ret void, !dbg !22
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @_Z2f8v() section "f" !dbg !23 {
entry:
  ret void, !dbg !24
}

!llvm.dbg.cu = !{!0, !3}
!llvm.ident = !{!5, !5}
!llvm.module.flags = !{!6, !7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 8.0.0 (trunk 346343) (llvm/trunk 346350)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None, rangesBaseAddress: true)
!1 = !DIFile(filename: "ranges.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch")
!2 = !{}
!3 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "clang version 8.0.0 (trunk 346343) (llvm/trunk 346350)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!4 = !DIFile(filename: "ranges_no_base.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch")
!5 = !{!"clang version 8.0.0 (trunk 346343) (llvm/trunk 346350)"}
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = distinct !DISubprogram(name: "f1", linkageName: "_Z2f1v", scope: !1, file: !1, line: 1, type: !10, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = !DILocation(line: 1, column: 42, scope: !9)
!13 = distinct !DISubprogram(name: "f2", linkageName: "_Z2f2v", scope: !1, file: !1, line: 2, type: !10, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!14 = !DILocation(line: 2, column: 42, scope: !13)
!15 = distinct !DISubprogram(name: "f4", linkageName: "_Z2f4v", scope: !1, file: !1, line: 4, type: !10, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!16 = !DILocation(line: 4, column: 42, scope: !15)
!17 = distinct !DISubprogram(name: "f5", linkageName: "_Z2f5v", scope: !1, file: !1, line: 5, type: !10, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!18 = !DILocation(line: 5, column: 42, scope: !17)
!19 = distinct !DISubprogram(name: "f6", linkageName: "_Z2f6v", scope: !1, file: !1, line: 6, type: !10, isLocal: false, isDefinition: true, scopeLine: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!20 = !DILocation(line: 6, column: 42, scope: !19)
!21 = distinct !DISubprogram(name: "f7", linkageName: "_Z2f7v", scope: !4, file: !4, line: 1, type: !10, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !3, retainedNodes: !2)
!22 = !DILocation(line: 1, column: 42, scope: !21)
!23 = distinct !DISubprogram(name: "f8", linkageName: "_Z2f8v", scope: !4, file: !4, line: 2, type: !10, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !3, retainedNodes: !2)
!24 = !DILocation(line: 2, column: 42, scope: !23)
