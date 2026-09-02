; RUN: llc -mtriple=x86_64-linux-gnu -enable-implicit-null-checks \
; RUN:   -stop-after=implicit-null-checks -o - %s | FileCheck %s

; ImplicitNullChecks must preserve the folded load's debug location on the
; resulting FAULTING_OP, otherwise the faulting instruction loses its source
; attribution in the DWARF line table.

; CHECK: [[LOC:![0-9]+]] = !DILocation(line: 4
; CHECK: FAULTING_OP {{.*}}debug-location [[LOC]] :: (load (s32) from %ir.p)

define i32 @f(ptr %p) !dbg !4 {
entry:
  %nn = icmp eq ptr %p, null
  br i1 %nn, label %null, label %notnull, !make.implicit !0

notnull:
  %v = load i32, ptr %p, !dbg !7
  ret i32 %v

null:
  unreachable
}

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!2}

!0 = !{}
!1 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, emissionKind: FullDebug)
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIFile(filename: "repro.c", directory: "/")
!4 = distinct !DISubprogram(name: "f", scope: !3, file: !3, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !1)
!5 = !DISubroutineType(types: !{})
!7 = !DILocation(line: 4, column: 10, scope: !4)
