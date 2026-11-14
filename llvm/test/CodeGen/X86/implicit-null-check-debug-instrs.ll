; RUN: llc -verify-machineinstrs -O3 -mtriple=x86_64-apple-macosx \
; RUN:   -enable-implicit-null-checks -imp-null-max-insts-to-consider=1 < %s \
; RUN:   | FileCheck %s

; Ensure a #dbg_value at the start of the non-null block does not consume the
; scan budget or prevent the load from becoming an implicit null check.

define i32 @ignore_debug_instructions_in_window(ptr %p) !dbg !4 {
; CHECK-LABEL: _ignore_debug_instructions_in_window:
; CHECK-NOT:   testq
; CHECK:       movl ({{.*}}), %eax ## on-fault:
entry:
  %isnull = icmp eq ptr %p, null
  br i1 %isnull, label %null, label %notnull, !make.implicit !0

null:
  ret i32 42

notnull:
    #dbg_value(ptr %p, !7, !DIExpression(), !8)
  %value = load i32, ptr %p, align 4
  ret i32 %value
}

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!2}

!0 = !{}
!1 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, emissionKind: FullDebug)
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIFile(filename: "implicit-null-check-debug-instrs.ll", directory: "/")
!4 = distinct !DISubprogram(name: "ignore_debug_instructions_in_window", scope: !3, file: !3, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !1)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !DILocalVariable(name: "p", arg: 1, scope: !4, file: !3, line: 1)
!8 = !DILocation(line: 2, column: 1, scope: !4)
