; REQUIRES: asserts
; RUN: opt -passes=elim-avail-extern -avail-extern-to-local -stats -S 2>&1 < %s | FileCheck %s
;
; RUN: echo '[{"Guid":1234, "Counters": [1]}]' | llvm-ctxprof-util fromYAML --input=- --output=%t_profile.ctxprofdata
;
; Because we pass a contextual profile with a root defined in this module, we expect the outcome to be the same as-if
; we passed -avail-extern-to-local, i.e. available_externally don't get elided and instead get converted to local linkage
; RUN: opt -passes='assign-guid,require<ctx-prof-analysis>,elim-avail-extern' -use-ctx-profile=%t_profile.ctxprofdata -stats -S 2>&1 < %s | FileCheck %s

; If the profile doesn't apply to this module, available_externally won't get converted to internal linkage, and will be
; removed instead.
; RUN: echo '[{"Guid":5678, "Counters": [1]}]' | llvm-ctxprof-util fromYAML --input=- --output=%t_profile_bad.ctxprofdata
; RUN: opt -passes='assign-guid,require<ctx-prof-analysis>,elim-avail-extern' -use-ctx-profile=%t_profile_bad.ctxprofdata -stats -S 2>&1 < %s | FileCheck %s --check-prefix=NOOP

declare void @call_out(ptr %fct)

define available_externally hidden void @f() {
  ret void
}

define available_externally hidden void @g() {
  ret void
}

define void @hello(ptr %g) !guid !0 {
  call void @f()
  %f = load ptr, ptr @f
  call void @call_out(ptr %f)
  ret void
}

!0 = !{i64 1234}

; CHECK: define internal void @f.__uniq.{{[0-9|a-f]*}}()
; CHECK: declare hidden void @g()
; CHECK: call void @f.__uniq.{{[0-9|a-f]*}}()
; CHECK-NEXT: load ptr, ptr @f
; CHECK-NEXT: call void @call_out(ptr %f)
; CHECK: Statistics Collected
; CHECK: 1 elim-avail-extern - Number of functions converted
; CHECK: 1 elim-avail-extern - Number of functions removed

; NOOP: 2 elim-avail-extern - Number of functions removed