; RUN: llc -mtriple=x86_64-linux-gnu %s -filetype=obj -o %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; Reconstruct this via clang and -O2.
; static void foo() {
;   struct X { int a; int b; } xyz;
; }

; int bar() {
;   foo();
;   return 1;
; }

; Check that we don't have the structure type for X since its scope has
; been optimized away.
; CHECK-NOT: DW_TAG_structure_type
; CHECK: DW_TAG_subprogram
; CHECK: DW_AT_name {{.*}}"bar"
; CHECK-NOT: DW_TAG_structure_type

; Function Attrs: nounwind readnone uwtable
define i32 @bar() #0 !dbg !4 {
entry:
  ret i32 1, !dbg !21
}

attributes #0 = { nounwind readnone uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18, !19}
!llvm.ident = !{!20}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5.0 (trunk 209255) (llvm/trunk 209253)", isOptimized: true, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "debug-dead-local-var.c", directory: "/usr/local/google/home/echristo")
!2 = !{}
!4 = distinct !DISubprogram(name: "bar", line: 11, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, unit: !0, scopeLine: 11, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "debug-dead-local-var.c", directory: "/usr/local/google/home/echristo")
!6 = !DISubroutineType(types: !7)
!7 = !{!8}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = distinct !DISubprogram(name: "foo", line: 6, isLocal: true, isDefinition: true, virtualIndex: 6, isOptimized: true, unit: !0, scopeLine: 6, file: !1, scope: !5, type: !10, variables: !12)
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = !{!13}
!13 = !DILocalVariable(name: "xyz", line: 8, scope: !9, file: !5, type: !14)
!14 = !DICompositeType(tag: DW_TAG_structure_type, name: "X", line: 8, size: 64, align: 32, file: !1, scope: !9, elements: !15)
!15 = !{!16, !17}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "a", line: 8, size: 32, align: 32, file: !1, scope: !14, baseType: !8)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "b", line: 8, size: 32, align: 32, offset: 32, file: !1, scope: !14, baseType: !8)
!18 = !{i32 2, !"Dwarf Version", i32 4}
!19 = !{i32 2, !"Debug Info Version", i32 3}
!20 = !{!"clang version 3.5.0 (trunk 209255) (llvm/trunk 209253)"}
!21 = !DILocation(line: 13, scope: !4)
