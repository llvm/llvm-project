; RUN: not llc -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs < %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=x86_64-unknown-linux-gnu -filetype=obj -verify-machineinstrs < %s -o /dev/null 2>&1 | FileCheck %s

;; Ensure that functions passing 7 or more integer arguments in GPRs (exceeding
;; the 6-GPR System V ABI limit) emit an error instead of reading out of bounds.

; CHECK: error: <unknown>:0:0: in function f1 i64 (i64, i64, i64, i64, i64, i64, i64): -fsanitize-kcfi-arity requires System V ABI compliance
define preserve_nonecc i64 @f1(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g) !kcfi_type !1 {
  %s1 = add i64 %a, %b
  %s2 = add i64 %c, %d
  %s3 = add i64 %e, %f
  %s4 = add i64 %s1, %s2
  %s5 = add i64 %s3, %g
  %r = add i64 %s4, %s5
  ret i64 %r
}

; CHECK: error: t.c:5:0: in function f2 i64 (i64, i64, i64, i64, i64, i64, i64, i64): -fsanitize-kcfi-arity requires System V ABI compliance
define preserve_nonecc i64 @f2(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, i64 %h) !dbg !5 !kcfi_type !1 {
  %s1 = add i64 %a, %b
  %s2 = add i64 %c, %d
  %s3 = add i64 %e, %f
  %s4 = add i64 %g, %h
  %s5 = add i64 %s1, %s2
  %s6 = add i64 %s3, %s4
  %r = add i64 %s5, %s6
  ret i64 %r
}

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!0, !2, !6}
!0 = !{i32 4, !"kcfi", i32 1}
!1 = !{i32 12345678}
!2 = !{i32 4, !"kcfi-arity", i32 1}
!3 = distinct !DICompileUnit(language: DW_LANG_C11, file: !4, isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = !DIFile(filename: "t.c", directory: "/")
!5 = distinct !DISubprogram(name: "f2", scope: !4, file: !4, line: 5, type: !7, scopeLine: 5, spFlags: DISPFlagDefinition, unit: !3)
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !DISubroutineType(types: !{})
