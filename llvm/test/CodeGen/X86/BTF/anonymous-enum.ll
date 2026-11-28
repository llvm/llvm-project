; RUN: llc -mtriple=x86_64-linux-gnu -filetype=asm -o - %s | FileCheck %s

; Verify compile-unit-only anonymous enums are emitted on x86_64.
;
; Source:
;   enum { false, true };
;   void f1(void) {}
; Compilation flag:
;   clang -target x86_64-linux-gnu -g -gbtf -S -emit-llvm t.c

define dso_local void @f1() !dbg !7 {
  ret void, !dbg !10
}

; CHECK: .section        .BTF,"",@progbits
; CHECK: # BTF_KIND_ENUM(
; CHECK: .ascii  "false"
; CHECK: .ascii  "true"
; CHECK: .ascii  "local_false"
; CHECK: .ascii  "local_true"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "/tmp")
!2 = !{!11}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 4, !"BTF", i32 1}
!7 = distinct !DISubprogram(name: "f1", scope: !1, file: !1, line: 2, type: !8, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocation(line: 2, column: 16, scope: !7)
!11 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !1, line: 1, baseType: !13, size: 32, elements: !14)
!12 = !{!17}
!13 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!14 = !{!15, !16}
!15 = !DIEnumerator(name: "false", value: 0)
!16 = !DIEnumerator(name: "true", value: 1)
!17 = !DICompositeType(tag: DW_TAG_enumeration_type, scope: !7, file: !1, line: 2, baseType: !13, size: 32, elements: !18)
!18 = !{!19, !20}
!19 = !DIEnumerator(name: "local_false", value: 0)
!20 = !DIEnumerator(name: "local_true", value: 1)
