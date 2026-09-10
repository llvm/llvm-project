; Regression test: trace-ret must NOT create an escaping return-value spill alloca in a
; function whose inline asm uses/clobbers the x86 base pointer (RBX). The spill alloca's
; address is passed to __sanitizer_cov_trace_ret, so it escapes; under KASAN it is redzoned,
; forcing 32-byte frame realignment -> a base pointer (RBX), which the X86 backend then
; rejects against the asm's RBX use with "Interference usage of base pointer/frame pointer".
; For such functions the return is traced as a null pointer instead.

; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=3 -sanitizer-coverage-trace-ret -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Scalar return in a function whose inline asm clobbers rbx (cf. CPUID "=b", RDTSC "~{rbx}").
define i32 @ret_scalar_clobbers_rbx(i32 %x) #0 !dbg !8 {
entry:
  call void asm sideeffect "nop", "~{rbx},~{dirflag},~{fpsr},~{flags}"() #0, !dbg !12
  ret i32 %x, !dbg !12
}
; CHECK-LABEL: define i32 @ret_scalar_clobbers_rbx(i32 %x)
; CHECK-NOT:   alloca
; CHECK:       call void @__sanitizer_cov_trace_ret(i64 ptrtoint (ptr @ret_scalar_clobbers_rbx to i64), i32 0, ptr null, ptr null, i32 0)
; CHECK:       ret i32 %x

; Control: the same scalar return WITHOUT base-pointer asm still spills and traces normally.
define i32 @ret_scalar_plain(i32 %x) #0 !dbg !13 {
entry:
  ret i32 %x, !dbg !14
}
; CHECK-LABEL: define i32 @ret_scalar_plain(i32 %x)
; CHECK:       %[[SLOT:.*]] = alloca i32
; CHECK:       store i32 %x, ptr %[[SLOT]]
; CHECK:       call void @__sanitizer_cov_trace_ret(i64 ptrtoint (ptr @ret_scalar_plain to i64), i32 4, ptr %[[SLOT]], ptr null, i32 0)
; CHECK:       ret i32 %x

attributes #0 = { nounwind sanitize_address }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, isOptimized: true, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = distinct !DISubprogram(name: "ret_scalar_clobbers_rbx", scope: !1, file: !1, line: 1, type: !9, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{!5, !5}
!12 = !DILocation(line: 1, column: 1, scope: !8)
!13 = distinct !DISubprogram(name: "ret_scalar_plain", scope: !1, file: !1, line: 2, type: !9, unit: !0, retainedNodes: !2)
!14 = !DILocation(line: 2, column: 1, scope: !13)
