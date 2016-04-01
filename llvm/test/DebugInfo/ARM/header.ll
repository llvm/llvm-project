; RUN: llc -mtriple armv7-apple-darwin < %s  | FileCheck %s

; Test that we don't pollute the start of the file with debug sections.
; This is particularly important on ARM MachO as a change in section order can
; cause a change the relaxation of the instructions used.

; CHECK:      .section        __TEXT,__text,regular,pure_instructions
; CHECK-NEXT: .syntax unified
; CHECK-NEXT: .globl  _f
; CHECK-NEXT: .p2align  2
; CHECK-NEXT: _f:                    @ @f

; CHECK:  .section        __DWARF,__debug_str,regular,debug

define void @f() !dbg !4 {
  ret void, !dbg !9
}
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "foo", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "/foo/test.c", directory: "/foo")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, variables: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !DILocation(line: 1, column: 15, scope: !4)
