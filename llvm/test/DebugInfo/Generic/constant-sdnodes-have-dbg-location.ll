; RUN: llc -debug -dag-dump-verbose < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

; CHECK: t{{[0-9]+}}: i32 = Constant<-1>test.c:4:5

define i32 @main() !dbg !4 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  ret i32 -1, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "", isOptimized: false, subprograms: !3)
!1 = !DIFile(filename: "test.c", directory: "/home/user/clang-llvm/build")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 2, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, variables: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !DILocation(line: 4, column: 5, scope: !4)
