; Test basic address sanitizer instrumentation for Myriad.
;
; RUN: opt -asan -asan-module -S  < %s | FileCheck %s

target triple = "sparc-myriad-rtems"
target datalayout = "E-m:e-p:32:32-i64:64-f128:64-n32-S64"
; CHECK: @llvm.global_ctors = {{.*}}@asan.module_ctor

define i32 @test_load(i32* %a) sanitize_address {
; CHECK-LABEL: @test_load
; CHECK-NOT: load
; CHECK:   ptrtoint i32* %a to i32
; CHECK:   %[[LOAD_ADDR:[^ ]*]] = and i32 %{{.*}}, -1073741825
; CHECK:   lshr i32 %{{.*}}, 29
; CHECK:   icmp eq i32 %{{.*}}, 4
; CHECK:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}!prof ![[PROF:[0-9]+]]
;
; This block checks whether the shadow byte is 0.
; CHECK:   lshr i32 %[[LOAD_ADDR]], 5
; CHECK:   add i32 %{{.*}}, -1694498816
; CHECK:   %[[LOAD_SHADOW_PTR:[^ ]*]] = inttoptr
; CHECK:   %[[LOAD_SHADOW:[^ ]*]] = load i8, i8* %[[LOAD_SHADOW_PTR]]
; CHECK:   icmp ne i8
; CHECK:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}!prof ![[PROF:[0-9]+]]
;
; This block refines the shadow test.
; CHECK:   and i32 %[[LOAD_ADDR]], 31
; CHECK:   add i32 %{{.*}}, 3
; CHECK:   trunc i32 %{{.*}} to i8
; CHECK:   icmp sge i8 %{{.*}}, %[[LOAD_SHADOW]]
; CHECK:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
;
; The crash block reports the error.
; CHECK:   call void @__asan_report_load4(i32 %[[LOAD_ADDR]])
; CHECK:   unreachable
;
; The actual load.
; CHECK:   %tmp1 = load i32, i32* %a
; CHECK:   ret i32 %tmp1

entry:
  %tmp1 = load i32, i32* %a, align 4
  ret i32 %tmp1
}

define void @test_store(i32* %a) sanitize_address {
; CHECK-LABEL: @test_store
; CHECK-NOT: store
; CHECK:   ptrtoint i32* %a to i32
; CHECK:   %[[STORE_ADDR:[^ ]*]] = and i32 %{{.*}}, -1073741825
; CHECK:   lshr i32 %{{.*}}, 29
; CHECK:   icmp eq i32 %{{.*}}, 4
; CHECK:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}!prof ![[PROF:[0-9]+]]
;
; This block checks whether the shadow byte is 0.
; CHECK:   lshr i32 %[[STORE_ADDR]], 5
; CHECK:   add i32 %{{.*}}, -1694498816
; CHECK:   %[[STORE_SHADOW_PTR:[^ ]*]] = inttoptr
; CHECK:   %[[STORE_SHADOW:[^ ]*]] = load i8, i8* %[[STORE_SHADOW_PTR]]
; CHECK:   icmp ne i8
; CHECK:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
;
; This block refines the shadow test.
; CHECK:   and i32 %[[STORE_ADDR]], 31
; CHECK:   add i32 %{{.*}}, 3
; CHECK:   trunc i32 %{{.*}} to i8
; CHECK:   icmp sge i8 %{{.*}}, %[[STORE_SHADOW]]
; CHECK:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
;
; The crash block reports the error.
; CHECK:   call void @__asan_report_store4(i32 %[[STORE_ADDR]])
; CHECK:   unreachable
; The actual store.
; CHECK:   store i32 42, i32* %a
; CHECK:   ret void
;

entry:
  store i32 42, i32* %a, align 4
  ret void
}

; CHECK: define internal void @asan.module_ctor()
; CHECK: call void @__asan_init()
