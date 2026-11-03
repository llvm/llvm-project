; Unbalanced llvm.sanitizer.instrumentation.begin/end regions are reported as a
; warning by every sanitizer pass that implements regions, once per function
; and pass.

; RUN: opt < %s -passes=msan -msan-kernel=1 -S -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --implicit-check-not=warning:
; RUN: opt < %s -passes=asan -asan-kernel=1 -S -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --implicit-check-not=warning:
; RUN: opt < %s -passes=tsan -S -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --implicit-check-not=warning:
; RUN: opt < %s -passes=sancov-module -sanitizer-coverage-level=3 \
; RUN:   -sanitizer-coverage-trace-pc -S -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --implicit-check-not=warning:

; Several passes analyzing the same function each report it. (Clang
; deduplicates the reports; opt does not.)
; RUN: opt < %s -passes='sancov-module,msan,tsan,asan' -msan-kernel=1 \
; RUN:   -asan-kernel=1 -sanitizer-coverage-level=3 -sanitizer-coverage-trace-pc \
; RUN:   -S -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --check-prefix=MULTI --implicit-check-not=warning:
; Five malformed functions, reported by each of the four passes.
; MULTI-COUNT-20: warning: {{.*}}in function '{{unmatched_end|not_closed|bad_join|nested_join|loop}}'

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.sanitizer.instrumentation.begin()
declare void @llvm.sanitizer.instrumentation.end()

; Balanced, nested and conditional-but-balanced regions are fine.
define void @balanced(ptr %p, i1 %c) disable_sanitizer_instrumentation {
entry:
  call void @llvm.sanitizer.instrumentation.begin()
  call void @llvm.sanitizer.instrumentation.begin()
  store i32 0, ptr %p
  call void @llvm.sanitizer.instrumentation.end()
  call void @llvm.sanitizer.instrumentation.end()
  br i1 %c, label %then, label %exit

then:
  call void @llvm.sanitizer.instrumentation.begin()
  store i32 1, ptr %p
  call void @llvm.sanitizer.instrumentation.end()
  br label %exit

exit:
  ret void
}

; Leaving the function without returning while a region is open is fine: after
; a noreturn call, and when an exception unwinds out of the function.
declare void @panic() noreturn
declare void @may_throw()
declare i32 @__gxx_personality_v0(...)

define void @open_at_unreachable(ptr %p, i1 %c) disable_sanitizer_instrumentation {
entry:
  call void @llvm.sanitizer.instrumentation.begin()
  br i1 %c, label %die, label %ok

die:
  call void @panic()
  unreachable

ok:
  store i32 0, ptr %p
  call void @llvm.sanitizer.instrumentation.end()
  ret void
}

define void @open_at_resume(ptr %p) disable_sanitizer_instrumentation personality ptr @__gxx_personality_v0 {
entry:
  call void @llvm.sanitizer.instrumentation.begin()
  invoke void @may_throw()
          to label %cont unwind label %lpad

cont:
  store i32 0, ptr %p
  call void @llvm.sanitizer.instrumentation.end()
  ret void

lpad:
  %lp = landingpad { ptr, i32 }
          cleanup
  resume { ptr, i32 } %lp
}

; CHECK: warning: {{.*}}regions.c:3:5: in function 'unmatched_end': sanitizer instrumentation region end without a matching begin
define void @unmatched_end(ptr %p) disable_sanitizer_instrumentation !dbg !5 {
entry:
  store i32 0, ptr %p
  call void @llvm.sanitizer.instrumentation.end(), !dbg !8
  ret void
}

; CHECK: warning: in function 'not_closed': sanitizer instrumentation region is not closed before return
define void @not_closed(ptr %p) disable_sanitizer_instrumentation {
entry:
  call void @llvm.sanitizer.instrumentation.begin()
  store i32 0, ptr %p
  ret void
}

; The region is only entered on one side of the branch, so the join is blamed
; rather than the end() or the return that follow it.
; CHECK: warning: in function 'bad_join': control flow merges from paths with different sanitizer instrumentation region nesting; the code that follows is treated as outside the region
define void @bad_join(ptr %p, i1 %c) disable_sanitizer_instrumentation {
entry:
  br i1 %c, label %then, label %join

then:
  call void @llvm.sanitizer.instrumentation.begin()
  br label %join

join:
  store i32 0, ptr %p
  call void @llvm.sanitizer.instrumentation.end()
  ret void
}

; If all paths are inside a region but at different depths, the code that
; follows stays instrumented.
; CHECK: warning: in function 'nested_join': control flow merges from paths with different sanitizer instrumentation region nesting; the code that follows is treated as nested in the smaller number of regions
define void @nested_join(ptr %p, i1 %c) disable_sanitizer_instrumentation {
entry:
  call void @llvm.sanitizer.instrumentation.begin()
  br i1 %c, label %then, label %join

then:
  call void @llvm.sanitizer.instrumentation.begin()
  br label %join

join:
  store i32 0, ptr %p
  call void @llvm.sanitizer.instrumentation.end()
  ret void
}

; A begin() in a loop body without an end() makes the loop header a bad join.
; CHECK: warning: in function 'loop': control flow merges
define void @loop(ptr %p, i1 %c) disable_sanitizer_instrumentation {
entry:
  br label %header

header:
  store i32 0, ptr %p
  call void @llvm.sanitizer.instrumentation.begin()
  br i1 %c, label %header, label %exit

exit:
  call void @llvm.sanitizer.instrumentation.end()
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "regions.c", directory: "/tmp")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 7, !"Dwarf Version", i32 5}
!5 = distinct !DISubprogram(name: "unmatched_end", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 3, column: 5, scope: !5)
