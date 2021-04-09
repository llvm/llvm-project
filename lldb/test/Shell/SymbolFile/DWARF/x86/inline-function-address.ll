; REQUIRES: lld
; RUN: llc -mtriple x86_64-pc-linux %s -filetype=obj -o %t.o
; RUN: ld.lld %t.o %t.o -o %t
; "foo" is defined in both compilation units, but there should be only one meaningful debuginfo entry
; RUN: lldb-test symbols --find=function --name=foo --function-flags=full %t | FileCheck %s
; CHECK: Function: {{.*}} "foo"
; CHECK-NOT: Function: {{.*}} "foo"

$foo = comdat any
define void @foo() comdat !dbg !6 {
entry:
  ret void
}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !{}, imports: !{}, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "inline-function-address.h", directory: "")
!2 = !DIFile(filename: "inline-function-address.c", directory: "")
!3 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!4 = !DISubroutineType(types: !{})
!5 = !DISubprogram(name: "foo", file: !1, line: 12, type: !4, flags: DIFlagPrototyped, spFlags: 0)
!6 = distinct !DISubprogram(name: "foo", file: !1, line: 12, type: !4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !5)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{}
!7 = !{i32 7, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
