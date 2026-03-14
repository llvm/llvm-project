 ; The .bc file was generated from this source using llvm-as from 8.0 release.
 ; RUN: llvm-dis < %s.bc | FileCheck %s

define i32 @main() !dbg !5 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  ret i32 0, !dbg !9
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 8.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "mainsubprogram.c", directory: "/dir")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 1, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 6, type: !6, scopeLine: 6, virtualIndex: 6, flags: DIFlagPrototyped | DIFlagMainSubprogram, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
; CHECK: !5 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 6, type: !6, scopeLine: 6, virtualIndex: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !0, retainedNodes: !2)
!6 = !DISubroutineType(types: !7)
!7 = !{!8}
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DILocation(line: 7, scope: !5)
