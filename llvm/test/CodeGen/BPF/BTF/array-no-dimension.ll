; RUN: llc -mtriple=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
;
; Rust and other languages may emit array types with no DISubrange dimensions.
; Verify BTFDebug does not crash and emits a zero-length array.

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"

define fastcc i1 @test() !dbg !4 {
  ret i1 false
}

; CHECK:             .section        .BTF,"",@progbits
; CHECK:             BTF_KIND_ARRAY
; CHECK:             .long   0{{$}}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !1, producer: "rustc", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.rs", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !1, size: 128, align: 64, elements: !9)
!9 = !{!10}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "field", scope: !8, file: !1, baseType: !11, size: 64, align: 64)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 384, align: 64, elements: !2)
!13 = !DIBasicType(name: "usize", size: 64, encoding: DW_ATE_unsigned)
