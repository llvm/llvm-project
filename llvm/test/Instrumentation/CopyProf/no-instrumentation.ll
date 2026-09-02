; Tests that CopyProf skips functions that should not be instrumented.
;
; RUN: opt < %s -passes='function(copyprof)' -S | FileCheck %s
; RUN: opt < %s -passes='function(copyprof-stores)' -S | FileCheck %s --check-prefix=STORES

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

;; Tests that a function without copyprof attributes is not instrumented.
define void @no_attrs(ptr %this) {
entry:
  ret void
}
; CHECK-LABEL: define void @no_attrs(ptr %this)
; CHECK-NOT:     call void @__copyprof_
; CHECK:         ret void
; STORES-LABEL: define void @no_attrs(ptr %this)
; STORES-NOT:     call void @__copyprof_store_callback
; STORES:         ret void

;; Tests that instrumentation is skipped when disable_sanitizer_instrumentation is present.
define void @disabled_ctor(ptr %this) "copyprof-ctor"="8" disable_sanitizer_instrumentation {
entry:
  ret void
}
; CHECK-LABEL: define void @disabled_ctor(ptr %this)
; CHECK-NOT:     call void @__copyprof_
; CHECK:         ret void

;; Tests that store instrumentation is skipped when disable_sanitizer_instrumentation is present.
define void @disabled_ctor_with_store(ptr %this) "copyprof-ctor"="8" disable_sanitizer_instrumentation {
entry:
  store i32 0, ptr %this
  ret void
}
; CHECK-LABEL: define void @disabled_ctor_with_store(ptr %this)
; CHECK-NOT:     call void @__copyprof_
; CHECK:         ret void
; STORES-LABEL: define void @disabled_ctor_with_store(ptr %this)
; STORES-NOT:     call void @__copyprof_store_callback
; STORES:         store i32 0, ptr %this
; STORES-NEXT:    ret void

;; Tests that store instrumentation is skipped for regular functions when disable_sanitizer_instrumentation
;; is present.
define void @disabled_stores(ptr %a) disable_sanitizer_instrumentation {
entry:
  store i32 42, ptr %a
  ret void
}
; STORES-LABEL: define void @disabled_stores(ptr %a)
; STORES-NOT:     call void @__copyprof_store_callback
; STORES:         store i32 42, ptr %a
; STORES-NEXT:    ret void

;; Tests that instrumentation is skipped for naked functions.
define void @naked_ctor_with_store(ptr %this) "copyprof-ctor"="8" naked {
entry:
  store i32 0, ptr null
  ret void
}
; CHECK-LABEL: define void @naked_ctor_with_store(ptr %this)
; CHECK-NOT:     call void @__copyprof_
; CHECK:         ret void
; STORES-LABEL: define void @naked_ctor_with_store(ptr %this)
; STORES-NOT:     call void @__copyprof_store_callback
; STORES:         store i32 0, ptr null
; STORES-NEXT:    ret void

;; Tests that the module constructor itself is not instrumented by either pass.
define internal void @copyprof.module_ctor() {
entry:
  ret void
}
; CHECK-LABEL: define internal void @copyprof.module_ctor()
; CHECK-NOT:     call void @__copyprof_
; CHECK:         ret void
; STORES-LABEL: define internal void @copyprof.module_ctor()
; STORES-NOT:     call void @__copyprof_store_callback
; STORES:         ret void
