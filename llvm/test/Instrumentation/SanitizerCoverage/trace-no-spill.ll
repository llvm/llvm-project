; Neither mode ever creates memory to report a value from, on any target: a
; value that is not already in memory is reported as the value itself. Nothing
; escapes to the stack, so no frame is realigned and no base pointer is
; reserved on the instrumentation's account.
;
; This is what makes the modes portable. A function whose inline assembly claims
; the x86-64 base pointer (rbx) is the case that would otherwise fail to
; compile - a redzoned, realigned spill slot needs the very register the
; assembly uses - and it now needs no special handling: it is instrumented
; exactly like the function below it.
;
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=3 -sanitizer-coverage-trace-args -sanitizer-coverage-trace-ret -S | FileCheck %s

; CHECK-NOT: alloca

; An rdtsc-style clobber of rbx.
define i32 @clobbers_rbx(i32 %x) #0 !dbg !6 {
entry:
  call void asm sideeffect "nop", "~{rbx},~{dirflag},~{fpsr},~{flags}"() #0, !dbg !9
  ret i32 %x, !dbg !9
}
; CHECK-LABEL: define i32 @clobbers_rbx(
; CHECK: %[[A:[0-9]+]] = zext i32 %x to i64
; CHECK: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @clobbers_rbx to i64), i32 0, i32 4, i64 %[[A]], ptr null, i32 0)
; CHECK: %[[R:[0-9]+]] = zext i32 %x to i64
; CHECK: call void @__sanitizer_cov_trace_ret(i64 ptrtoint (ptr @clobbers_rbx to i64), i32 4, i64 %[[R]], ptr null, i32 0)
; CHECK: ret i32 %x

; A cpuid-style "=b" output operand names the same register.
define i32 @uses_b_constraint(i32 %x) #0 !dbg !10 {
entry:
  %0 = call i32 asm "cpuid", "=b,0"(i32 %x) #0, !dbg !11
  ret i32 %0, !dbg !11
}
; CHECK-LABEL: define i32 @uses_b_constraint(
; CHECK: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @uses_b_constraint to i64), i32 0, i32 4, i64 %{{[0-9]+}}, ptr null, i32 0)
; CHECK: call void @__sanitizer_cov_trace_ret(i64 ptrtoint (ptr @uses_b_constraint to i64), i32 4, i64 %{{[0-9]+}}, ptr null, i32 0)

; The same function without such inline assembly, instrumented identically.
define i32 @plain(i32 %x) #0 !dbg !12 {
entry:
  ret i32 %x, !dbg !13
}
; CHECK-LABEL: define i32 @plain(
; CHECK: %[[PA:[0-9]+]] = zext i32 %x to i64
; CHECK: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @plain to i64), i32 0, i32 4, i64 %[[PA]], ptr null, i32 0)
; CHECK: %[[PR:[0-9]+]] = zext i32 %x to i64
; CHECK: call void @__sanitizer_cov_trace_ret(i64 ptrtoint (ptr @plain to i64), i32 4, i64 %[[PR]], ptr null, i32 0)

attributes #0 = { nounwind sanitize_address }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1, !2}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, isOptimized: true, emissionKind: FullDebug)
!1 = !{i32 2, !"Dwarf Version", i32 5}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIFile(filename: "no-spill.c", directory: "/")
!4 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = !DISubroutineType(types: !{!4, !4})
!6 = distinct !DISubprogram(name: "clobbers_rbx", scope: !3, file: !3, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !20)
!9 = !DILocation(line: 1, column: 1, scope: !6)
!10 = distinct !DISubprogram(name: "uses_b_constraint", scope: !3, file: !3, line: 5, type: !5, scopeLine: 5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !20)
!11 = !DILocation(line: 5, column: 1, scope: !10)
!12 = distinct !DISubprogram(name: "plain", scope: !3, file: !3, line: 9, type: !5, scopeLine: 9, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !20)
!13 = !DILocation(line: 9, column: 1, scope: !12)
!20 = !{}
