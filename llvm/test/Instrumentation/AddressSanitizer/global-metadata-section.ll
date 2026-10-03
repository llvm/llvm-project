; Test that the -asan-globals-metadata-section option works as expected
;
; RUN: opt < %s -passes=asan -asan-globals-metadata-section=my_section -S | FileCheck %s
target triple = "x86_64-unknown-linux-gnu"

@global = global i32 0, align 4

; CHECK: @__asan_global_global = {{.*}} section "my_section"{{.*}}
; CHECK: @llvm.compiler.used = {{.*}}ptr @__asan_global_global{{.*}}

; CHECK: define internal void @asan.module_ctor()
; CHECK-NOT: __asan_register
; CHECK: ret void
