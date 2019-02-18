; RUN: %llc_dwarf %s -filetype=obj -o %t
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s
; REQUIRES: default_triple

; Reduced test case from PR35212. Two DISubprogram belong to a different CU but
; share a scope. Both are declarations and end up in the scope's CU. We want to
; check that the CU from the context DIE is used (rather than from the IR).
; This manifests itself by the DW_base_type ending up in the second CU, rather
; than in the first one as specified in the IR.

; CHECK: DW_TAG_compile_unit
; CHECK-NEXT: discriminator 0
; CHECK: DW_TAG_compile_unit
; CHECK-NEXT: discriminator 1
; CHECK: DW_TAG_structure_type
; CHECK-NOT: NULL
; CHECK: DW_TAG_subprogram
; CHECK-NOT: NULL
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: NULL
; CHECK: DW_AT_type{{.*}}"usize"
; CHECK: NULL
; CHECK: DW_TAG_base_type
; CHECK-NOT: NULL
; CHECK: DW_AT_name{{.*}}"usize"

define hidden void @foo() !dbg !4 {
  ret void, !dbg !7
}

!llvm.dbg.cu = !{!0, !2}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !1, producer: "clang LLVM (rustc version 1.23.0-nightly (discriminator 0))", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "../lib.rs", directory: "/home/alex/code/rust4/lol")
!2 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !1, producer: "clang LLVM (rustc version 1.23.0-nightly (discriminator 1))", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "clone<alloc::string::String>", linkageName: "_ZN5alloc3vec8{{impl}}28clone<alloc::string::String>E", scope: null, file: !1, line: 1519, type: !5, isLocal: false, isDefinition: true, scopeLine: 1519, flags: DIFlagPrototyped, isOptimized: true, unit: !0, templateParams: !6, retainedNodes: !6)
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 1612, scope: !8, inlinedAt: !11)
!8 = distinct !DILexicalBlock(scope: !9, file: !1, line: 86, column: 12)
!9 = distinct !DISubprogram(name: "allocate_in<alloc::string::String,alloc::heap::Heap>", linkageName: "_ZN5alloc7raw_vec8{{impl}}52allocate_in<alloc::string::String,alloc::heap::Heap>E", scope: !10, file: !1, line: 82, type: !5, isLocal: false, isDefinition: true, scopeLine: 82, flags: DIFlagPrototyped, isOptimized: true, unit: !2, templateParams: !6, retainedNodes: !6)
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "RawVec<alloc::string::String, alloc::heap::Heap>", file: !1, size: 128, align: 64, elements: !6, identifier: "5c6e4db16d2c64555e40661d70c4d81e")
!11 = distinct !DILocation(line: 86, scope: !8, inlinedAt: !12)
!12 = distinct !DILocation(line: 141, scope: !13, inlinedAt: !17)
!13 = distinct !DISubprogram(name: "with_capacity<alloc::string::String>", linkageName: "_ZN5alloc7raw_vec8{{impl}}36with_capacity<alloc::string::String>E", scope: !10, file: !1, line: 140, type: !5, isLocal: false, isDefinition: true, scopeLine: 140, flags: DIFlagPrototyped, isOptimized: true, unit: !0, templateParams: !6, retainedNodes: !14)
!14 = !{!15}
!15 = !DILocalVariable(name: "cap", arg: 1, scope: !13, file: !1, line: 1, type: !16)
!16 = !DIBasicType(name: "usize", size: 64, encoding: DW_ATE_unsigned)
!17 = !DILocation(line: 1521, scope: !4)
