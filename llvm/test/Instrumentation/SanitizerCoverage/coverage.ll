
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

; CHECK0-NOT: @llvm.global_ctors = {{.*}}{ i32 2, ptr @sancov.module_ctor }
; CHECK1: @llvm.global_ctors = {{.*}}{ i32 2, ptr @sancov.module_ctor, ptr null }
; CHECK2: @llvm.global_ctors = {{.*}}{ i32 2, ptr @sancov.module_ctor, ptr null }

; CHECK0-NOT: call void @__sanitizer_cov(
; CHECK0-NOT: call void @__sanitizer_cov_module_init(

; CHECK1-LABEL: define void @foo
; CHECK1: %0 = load atomic i32, ptr {{.*}} monotonic, align 4, !nosanitize
; CHECK1: %1 = icmp sge i32 0, %0
; CHECK1: br i1 %1, label %2, label %3
; CHECK1: call void @__sanitizer_cov(ptr{{.*}})
; CHECK1: call void asm sideeffect "", ""()
; CHECK1-NOT: call void @__sanitizer_cov
; CHECK1: ret void

; CHECK1-LABEL: define internal void @sancov.module_ctor
; CHECK1-NOT: ret
; CHECK1: call void @__sanitizer_cov_module_init({{.*}}, i64 2, {{.*}}@__sancov_gen_modname
; CHECK1: ret

; CHECK_WITH_CHECK-LABEL: define void @foo
; CHECK_WITH_CHECK: __sanitizer_cov_with_check
; CHECK_WITH_CHECK: ret void
; CHECK_WITH_CHECK-LABEL: define internal void @sancov.module_ctor
; CHECK_WITH_CHECK-NOT: ret
; CHECK_WITH_CHECK: call void @__sanitizer_cov_module_init({{.*}}, i64 3,
; CHECK_WITH_CHECK: ret

; CHECK2-LABEL: define void @foo
; CHECK2: call void @__sanitizer_cov
; CHECK2: call void asm sideeffect "", ""()
; CHECK2: call void @__sanitizer_cov
; CHECK2: call void asm sideeffect "", ""()
; CHECK2: call void asm sideeffect "", ""()
; CHECK2-NOT: call void @__sanitizer_cov
; CHECK2: ret void

; CHECK2-LABEL: define internal void @sancov.module_ctor
; CHECK2-NOT: ret
; CHECK2: call void @__sanitizer_cov_module_init({{.*}}, i64 3,
; CHECK2: ret

; CHECK3-LABEL: define void @foo
; CHECK3: call void @__sanitizer_cov
; CHECK3: call void @__sanitizer_cov
; CHECK3-NOT: ret void
; CHECK3: call void @__sanitizer_cov
; CHECK3-NOT: call void @__sanitizer_cov
; CHECK3: ret void

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
