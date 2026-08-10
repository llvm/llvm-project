
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=4 -sanitizer-coverage-trace-pc  -S | FileCheck %s --check-prefix=CHECK_TRACE_PC
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=3 -sanitizer-coverage-prune-blocks=1 -S | FileCheck %s --check-prefix=CHECKPRUNE

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"
define void @foo(ptr %a) sanitize_address {
entry:
  %tobool = icmp eq ptr %a, null
  br i1 %tobool, label %if.end, label %if.then

  if.then:                                          ; preds = %entry
  store i32 0, ptr %a, align 4
  br label %if.end

  if.end:                                           ; preds = %entry, %if.then
  ret void
}

%struct.StructWithVptr = type { ptr }

define void @CallViaVptr(ptr %foo) uwtable sanitize_address {
entry:
  %vtable = load ptr, ptr %foo, align 8
  %0 = load ptr, ptr %vtable, align 8
  tail call void %0(ptr %foo)
  tail call void %0(ptr %foo)
  tail call void asm sideeffect "", ""()
  ret void
}

; CHECK_TRACE_PC-LABEL: define void @foo
; CHECK_TRACE_PC: call void @__sanitizer_cov_trace_pc
; CHECK_TRACE_PC: call void @__sanitizer_cov_trace_pc
; CHECK_TRACE_PC: call void @__sanitizer_cov_trace_pc
; CHECK_TRACE_PC: ret void

; CHECK_TRACE_PC-LABEL: define void @CallViaVptr
; CHECK_TRACE_PC: call void @__sanitizer_cov_trace_pc_indir
; CHECK_TRACE_PC: call void @__sanitizer_cov_trace_pc_indir
; CHECK_TRACE_PC: ret void

define void @call_unreachable() uwtable sanitize_address {
entry:
  unreachable
}

; CHECKPRUNE-LABEL: define void @foo
; CHECKPRUNE: call void @__sanitizer_cov
; CHECKPRUNE: call void @__sanitizer_cov
; CHECKPRUNE: call void @__sanitizer_cov
; CHECKPRUNE-NOT: call void @__sanitizer_cov
; CHECKPRUNE: ret void
