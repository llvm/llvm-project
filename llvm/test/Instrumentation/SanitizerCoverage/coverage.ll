; RUN: opt < %s -sancov -sanitizer-coverage-level=0 -S | FileCheck %s --check-prefix=CHECK0
; RUN: opt < %s -sancov -sanitizer-coverage-level=1 -S | FileCheck %s --check-prefix=CHECK1
; RUN: opt < %s -sancov -sanitizer-coverage-level=2 -S | FileCheck %s --check-prefix=CHECK2
; RUN: opt < %s -sancov -sanitizer-coverage-level=2 -sanitizer-coverage-block-threshold=10 -S | FileCheck %s --check-prefix=CHECK2
; RUN: opt < %s -sancov -sanitizer-coverage-level=2 -sanitizer-coverage-block-threshold=0  -S | FileCheck %s --check-prefix=CHECK_WITH_CHECK
; RUN: opt < %s -sancov -sanitizer-coverage-level=2 -sanitizer-coverage-block-threshold=1  -S | FileCheck %s --check-prefix=CHECK_WITH_CHECK
; RUN: opt < %s -sancov -sanitizer-coverage-level=3 -sanitizer-coverage-block-threshold=10 -S | FileCheck %s --check-prefix=CHECK3
; RUN: opt < %s -sancov -sanitizer-coverage-level=4 -S | FileCheck %s --check-prefix=CHECK4
; RUN: opt < %s -sancov -sanitizer-coverage-level=4 -sanitizer-coverage-trace-pc  -S | FileCheck %s --check-prefix=CHECK_TRACE_PC_INDIR
; RUN: opt < %s -sancov -sanitizer-coverage-level=3 -sanitizer-coverage-8bit-counters=1  -S | FileCheck %s --check-prefix=CHECK-8BIT

; RUN: opt < %s -sancov -sanitizer-coverage-level=2 -sanitizer-coverage-block-threshold=10 \
; RUN:      -S | FileCheck %s --check-prefix=CHECK2
; RUN: opt < %s -sancov -sanitizer-coverage-level=2 -sanitizer-coverage-block-threshold=1 \
; RUN:      -S | FileCheck %s --check-prefix=CHECK_WITH_CHECK
; RUN: opt < %s -sancov -sanitizer-coverage-level=3 -sanitizer-coverage-prune-blocks=1 -S | FileCheck %s --check-prefix=CHECKPRUNE

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"
define void @foo(i32* %a) sanitize_address {
entry:
  %tobool = icmp eq i32* %a, null
  br i1 %tobool, label %if.end, label %if.then

  if.then:                                          ; preds = %entry
  store i32 0, i32* %a, align 4
  br label %if.end

  if.end:                                           ; preds = %entry, %if.then
  ret void
}

; CHECK0-NOT: @llvm.global_ctors = {{.*}}{ i32 2, void ()* @sancov.module_ctor }
; CHECK1: @llvm.global_ctors = {{.*}}{ i32 2, void ()* @sancov.module_ctor, i8* null }
; CHECK2: @llvm.global_ctors = {{.*}}{ i32 2, void ()* @sancov.module_ctor, i8* null }

; CHECK0-NOT: call void @__sanitizer_cov(
; CHECK0-NOT: call void @__sanitizer_cov_module_init(

; CHECK1-LABEL: define void @foo
; CHECK1: %0 = load atomic i32, i32* {{.*}} monotonic, align 4, !nosanitize
; CHECK1: %1 = icmp sge i32 0, %0
; CHECK1: br i1 %1, label %2, label %3
; CHECK1: call void @__sanitizer_cov(i32*{{.*}})
; CHECK1: call void asm sideeffect "", ""()
; CHECK1-NOT: call void @__sanitizer_cov
; CHECK1: ret void

; CHECK1-LABEL: define internal void @sancov.module_ctor
; CHECK1-NOT: ret
; CHECK1: call void @__sanitizer_cov_module_init({{.*}}, i64 2,
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

; test -sanitizer-coverage-8bit-counters=1
; CHECK-8BIT-LABEL: define void @foo

; CHECK-8BIT: [[V11:%[0-9]*]] = load i8{{.*}}!nosanitize
; CHECK-8BIT: [[V12:%[0-9]*]] = add i8 [[V11]], 1
; CHECK-8BIT: store i8 [[V12]]{{.*}}!nosanitize
; CHECK-8BIT: [[V21:%[0-9]*]] = load i8{{.*}}!nosanitize
; CHECK-8BIT: [[V22:%[0-9]*]] = add i8 [[V21]], 1
; CHECK-8BIT: store i8 [[V22]]{{.*}}!nosanitize
; CHECK-8BIT: [[V31:%[0-9]*]] = load i8{{.*}}!nosanitize
; CHECK-8BIT: [[V32:%[0-9]*]] = add i8 [[V31]], 1
; CHECK-8BIT: store i8 [[V32]]{{.*}}!nosanitize
; CHECK-8BIT: [[V41:%[0-9]*]] = load i8{{.*}}!nosanitize
; CHECK-8BIT: [[V42:%[0-9]*]] = add i8 [[V41]], 1
; CHECK-8BIT: store i8 [[V42]]{{.*}}!nosanitize

; CHECK-8BIT: ret void


%struct.StructWithVptr = type { i32 (...)** }

define void @CallViaVptr(%struct.StructWithVptr* %foo) uwtable sanitize_address {
entry:
  %0 = bitcast %struct.StructWithVptr* %foo to void (%struct.StructWithVptr*)***
  %vtable = load void (%struct.StructWithVptr*)**, void (%struct.StructWithVptr*)*** %0, align 8
  %1 = load void (%struct.StructWithVptr*)*, void (%struct.StructWithVptr*)** %vtable, align 8
  tail call void %1(%struct.StructWithVptr* %foo)
  tail call void %1(%struct.StructWithVptr* %foo)
  tail call void asm sideeffect "", ""()
  ret void
}

; We expect to see two calls to __sanitizer_cov_indir_call16
; with different values of second argument.
; CHECK4-LABEL: define void @CallViaVptr
; CHECK4: call void @__sanitizer_cov_indir_call16({{.*}},[[CACHE:.*]])
; CHECK4-NOT: call void @__sanitizer_cov_indir_call16({{.*}},[[CACHE]])
; CHECK4: ret void

; CHECK_TRACE_PC_INDIR-LABEL: define void @CallViaVptr
; CHECK_TRACE_PC_INDIR: call void @__sanitizer_cov_trace_pc_indir
; CHECK_TRACE_PC_INDIR: call void @__sanitizer_cov_trace_pc_indir
; CHECK_TRACE_PC_INDIR: ret void

define void @call_unreachable() uwtable sanitize_address {
entry:
  unreachable
}

; CHECK4-LABEL: define void @call_unreachable
; CHECK4-NOT: __sanitizer_cov
; CHECK4: unreachable

; CHECKPRUNE-LABEL: define void @foo
; CHECKPRUNE: call void @__sanitizer_cov
; CHECKPRUNE: call void @__sanitizer_cov
; CHECKPRUNE: call void @__sanitizer_cov
; CHECKPRUNE-NOT: call void @__sanitizer_cov
; CHECKPRUNE: ret void
