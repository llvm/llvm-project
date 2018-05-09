; RUN: llc -debug -dag-dump-verbose < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

; CHECK: t{{[0-9]+}}: f64 = ConstantFP<1.500000e+00>test.c:3:5

define double @f() !dbg !4 {
entry:
  ret double 1.500000e+00, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "", isOptimized: false, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/home/user/clang-llvm/build")
!2 = !{}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIBasicType(name: "double", size: 64, align: 64, encoding: DW_ATE_float)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !DILocation(line: 3, column: 5, scope: !4)
