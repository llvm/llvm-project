; Test basic address sanitizer instrumentation.
;
; RUN: opt < %s -asan -asan-module -S | FileCheck --check-prefixes=CHECK,CHECK-S3 %s
; RUN: opt < %s -asan -asan-module -asan-mapping-scale=5 -S | FileCheck --check-prefixes=CHECK,CHECK-S5 %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"
; CHECK: @llvm.global_ctors = {{.*}}@asan.module_ctor

define i32 @test_load(i32* %a) sanitize_address {
; CHECK-LABEL: @test_load
; CHECK-NOT: load
; CHECK:   %[[LOAD_ADDR:[^ ]*]] = ptrtoint i32* %a to i64
; CHECK-S3:   lshr i64 %[[LOAD_ADDR]], 3
; CHECK-S5:   lshr i64 %[[LOAD_ADDR]], 5
; CHECK:   {{or|add}}
; CHECK:   %[[LOAD_SHADOW_PTR:[^ ]*]] = inttoptr
; CHECK:   %[[LOAD_SHADOW:[^ ]*]] = load i8, i8* %[[LOAD_SHADOW_PTR]]
; CHECK:   icmp ne i8
; CHECK:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}!prof ![[PROF:[0-9]+]]
;
; First instrumentation block refines the shadow test.
; CHECK-S3:   and i64 %[[LOAD_ADDR]], 7
; CHECK-S5:   and i64 %[[LOAD_ADDR]], 31
; CHECK:   add i64 %{{.*}}, 3
; CHECK:   trunc i64 %{{.*}} to i8
; CHECK:   icmp sge i8 %{{.*}}, %[[LOAD_SHADOW]]
; CHECK:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
;
; The crash block reports the error.
; CHECK:   call void @__asan_report_load4(i64 %[[LOAD_ADDR]])
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
; CHECK:   %[[STORE_ADDR:[^ ]*]] = ptrtoint i32* %a to i64
; CHECK-S3:   lshr i64 %[[STORE_ADDR]], 3
; CHECK-S5:   lshr i64 %[[STORE_ADDR]], 5
; CHECK:   {{or|add}}
; CHECK:   %[[STORE_SHADOW_PTR:[^ ]*]] = inttoptr
; CHECK:   %[[STORE_SHADOW:[^ ]*]] = load i8, i8* %[[STORE_SHADOW_PTR]]
; CHECK:   icmp ne i8
; CHECK:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
;
; First instrumentation block refines the shadow test.
; CHECK-S3:   and i64 %[[STORE_ADDR]], 7
; CHECK-S5:   and i64 %[[STORE_ADDR]], 31
; CHECK:   add i64 %{{.*}}, 3
; CHECK:   trunc i64 %{{.*}} to i8
; CHECK:   icmp sge i8 %{{.*}}, %[[STORE_SHADOW]]
; CHECK:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
;
; The crash block reports the error.
; CHECK:   call void @__asan_report_store4(i64 %[[STORE_ADDR]])
; CHECK:   unreachable
;
; The actual load.
; CHECK:   store i32 42, i32* %a
; CHECK:   ret void
;

entry:
  store i32 42, i32* %a, align 4
  ret void
}

; Check that asan leaves just one alloca.

declare void @alloca_test_use([10 x i8]*)
define void @alloca_test() sanitize_address {
entry:
  %x = alloca [10 x i8], align 1
  %y = alloca [10 x i8], align 1
  %z = alloca [10 x i8], align 1
  call void @alloca_test_use([10 x i8]* %x)
  call void @alloca_test_use([10 x i8]* %y)
  call void @alloca_test_use([10 x i8]* %z)
  ret void
}

; CHECK-LABEL: define void @alloca_test()
; CHECK: %asan_local_stack_base = alloca
; CHECK: = alloca
; CHECK-NOT: = alloca
; CHECK: ret void

define void @LongDoubleTest(x86_fp80* nocapture %a) nounwind uwtable sanitize_address {
entry:
    store x86_fp80 0xK3FFF8000000000000000, x86_fp80* %a, align 16
    ret void
}

; CHECK-LABEL: LongDoubleTest
; CHECK: __asan_report_store_n
; CHECK: __asan_report_store_n
; CHECK: ret void


define void @i40test(i40* %a, i40* %b) nounwind uwtable sanitize_address {
  entry:
  %t = load i40, i40* %a
  store i40 %t, i40* %b, align 8
  ret void
}

; CHECK-LABEL: i40test
; CHECK: __asan_report_load_n{{.*}}, i64 5)
; CHECK: __asan_report_load_n{{.*}}, i64 5)
; CHECK: __asan_report_store_n{{.*}}, i64 5)
; CHECK: __asan_report_store_n{{.*}}, i64 5)
; CHECK: ret void

define void @i64test_align1(i64* %b) nounwind uwtable sanitize_address {
  entry:
  store i64 0, i64* %b, align 1
  ret void
}

; CHECK-LABEL: i64test_align1
; CHECK: __asan_report_store_n{{.*}}, i64 8)
; CHECK: __asan_report_store_n{{.*}}, i64 8)
; CHECK: ret void


define void @i80test(i80* %a, i80* %b) nounwind uwtable sanitize_address {
  entry:
  %t = load i80, i80* %a
  store i80 %t, i80* %b, align 8
  ret void
}

; CHECK-LABEL: i80test
; CHECK: __asan_report_load_n{{.*}}, i64 10)
; CHECK: __asan_report_load_n{{.*}}, i64 10)
; CHECK: __asan_report_store_n{{.*}}, i64 10)
; CHECK: __asan_report_store_n{{.*}}, i64 10)
; CHECK: ret void

; asan should not instrument functions with available_externally linkage.
define available_externally i32 @f_available_externally(i32* %a) sanitize_address  {
entry:
  %tmp1 = load i32, i32* %a
  ret i32 %tmp1
}
; CHECK-LABEL: @f_available_externally
; CHECK-NOT: __asan_report
; CHECK: ret i32

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) nounwind
declare void @llvm.memmove.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) nounwind

define void @memintr_test(i8* %a, i8* %b) nounwind uwtable sanitize_address {
  entry:
  tail call void @llvm.memset.p0i8.i64(i8* %a, i8 0, i64 100, i32 1, i1 false)
  tail call void @llvm.memmove.p0i8.p0i8.i64(i8* %a, i8* %b, i64 100, i32 1, i1 false)
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %a, i8* %b, i64 100, i32 1, i1 false)
  ret void
}

; CHECK-LABEL: memintr_test
; CHECK: __asan_memset
; CHECK: __asan_memmove
; CHECK: __asan_memcpy
; CHECK: ret void

declare void @llvm.memset.element.unordered.atomic.p0i8.i64(i8* nocapture writeonly, i8, i64, i32) nounwind
declare void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32) nounwind
declare void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32) nounwind

define void @memintr_element_atomic_test(i8* %a, i8* %b) nounwind uwtable sanitize_address {
  ; This is a canary test to make sure that these don't get lowered into calls that don't
  ; have the element-atomic property. Eventually, asan will have to be enhanced to lower
  ; these properly.
  ; CHECK-LABEL: memintr_element_atomic_test
  ; CHECK-NEXT: tail call void @llvm.memset.element.unordered.atomic.p0i8.i64(i8* align 1 %a, i8 0, i64 100, i32 1)
  ; CHECK-NEXT: tail call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i64(i8* align 1 %a, i8* align 1 %b, i64 100, i32 1)
  ; CHECK-NEXT: tail call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i64(i8* align 1 %a, i8* align 1 %b, i64 100, i32 1)
  ; CHECK-NEXT: ret void
  tail call void @llvm.memset.element.unordered.atomic.p0i8.i64(i8* align 1 %a, i8 0, i64 100, i32 1)
  tail call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i64(i8* align 1 %a, i8* align 1 %b, i64 100, i32 1)
  tail call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i64(i8* align 1 %a, i8* align 1 %b, i64 100, i32 1)
  ret void
}


; CHECK-LABEL: @test_swifterror
; CHECK-NOT: __asan_report_load
; CHECK: ret void
define void @test_swifterror(i8** swifterror) sanitize_address {
  %swifterror_ptr_value = load i8*, i8** %0
  ret void
}

; CHECK-LABEL: @test_swifterror_2
; CHECK-NOT: __asan_report_store
; CHECK: ret void
define void @test_swifterror_2(i8** swifterror) sanitize_address {
  store i8* null, i8** %0
  ret void
}

; CHECK-LABEL: @test_swifterror_3
; CHECK-NOT: __asan_report_store
; CHECK: ret void
define void @test_swifterror_3() sanitize_address {
  %swifterror_addr = alloca swifterror i8*
  store i8* null, i8** %swifterror_addr
  call void @test_swifterror_2(i8** swifterror %swifterror_addr)
  ret void
}

; CHECK: define internal void @asan.module_ctor()
; CHECK: call void @__asan_init()

; PROF
; CHECK: ![[PROF]] = !{!"branch_weights", i32 1, i32 100000}
