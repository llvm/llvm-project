; RUN: llc -mtriple=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -mtriple=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Rust's char type uses DW_ATE_UTF encoding. Verify BTF treats it as an
; unsigned integer (BTF_KIND_INT with encoding 0).

; Source (Rust):
;   struct Option { val: char }

define void @test(i32 %0) !dbg !4 {
  ret void
}

; CHECK:             .section        .BTF,"",@progbits

; The "char" type should appear as BTF_KIND_INT with encoding=0 (unsigned),
; size=4.
; CHECK:             .long   [[CHAR:[0-9]+]]         # BTF_KIND_INT(id = {{[0-9]+}})
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   32                      # 0x20

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !1, producer: "rustc", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false)
!1 = !DIFile(filename: "test.rs", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "test", scope: null, file: !1, line: 1, type: !5, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !7)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{!8}
!8 = !DILocalVariable(name: "c", arg: 1, scope: !4, file: !1, line: 1, type: !9)
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Option", file: !1, size: 32, align: 32, elements: !10)
!10 = !{!11}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "val", scope: !9, file: !1, baseType: !12, size: 32, align: 32)
!12 = !DIBasicType(name: "char", size: 32, encoding: DW_ATE_UTF)
