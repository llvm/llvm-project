; RUN: llc -mtriple=x86_64-linux-gnu -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Verify that BTF and DWARF coexist: both .BTF and .debug_info sections
; are emitted when the BTF module flag is set alongside DWARF metadata.
;
; Source:
;   void f1(void) {}
; Compilation flag:
;   clang -target x86_64-linux-gnu -g -gbtf -S -emit-llvm t.c

define dso_local void @f1() !dbg !7 {
  ret void, !dbg !10
}

; Both DWARF and BTF sections must be present.
; CHECK-DAG:         .section        .debug_info
; CHECK-DAG:         .section        .BTF,"",@progbits
; CHECK-DAG:         .section        .BTF.ext,"",@progbits

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 4, !"BTF", i32 1}
!7 = distinct !DISubprogram(name: "f1", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocation(line: 1, column: 16, scope: !7)
