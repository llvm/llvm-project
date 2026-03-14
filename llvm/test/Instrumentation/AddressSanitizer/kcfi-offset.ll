;; Test that we set patchable-function-prefix for asan.module_ctor when kcfi-offset is defined.

; RUN: opt < %s -passes=asan -S | FileCheck %s

; CHECK: @llvm.global_ctors = {{.*}}{ i32 1, ptr @asan.module_ctor, ptr @asan.module_ctor }

; CHECK: define internal void @asan.module_ctor()
; CHECK-SAME: #[[#ATTR:]]
; CHECK-SAME: !kcfi_type

; CHECK: attributes #[[#ATTR]] = { {{.*}} "patchable-function-prefix"="3" }

!llvm.module.flags = !{!0, !1}
!0 = !{i32 4, !"kcfi", i32 1}
!1 = !{i32 4, !"kcfi-offset", i32 3}
