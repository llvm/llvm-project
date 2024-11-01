; Test basic address sanitizer instrumentation.
;

; RUN: opt < %s -passes=asan -S | FileCheck --check-prefixes=CHECK,CHECK-S3 %s
; RUN: opt < %s -passes=asan -asan-mapping-scale=5 -S | FileCheck --check-prefixes=CHECK,CHECK-S5 %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"
; CHECK: @llvm.used = appending global [1 x ptr] [ptr @asan.module_ctor]
; CHECK: @llvm.global_ctors = {{.*}}{ i32 1, ptr @asan.module_ctor, ptr @asan.module_ctor }

define i32 @test_load(ptr %a) sanitize_address {
; CHECK-LABEL: @test_load
; CHECK-NOT: load
; CHECK:   %[[LOAD_ADDR:[^ ]*]] = ptrtoint ptr %a to i64
; CHECK-S3:   lshr i64 %[[LOAD_ADDR]], 3
; CHECK-S5:   lshr i64 %[[LOAD_ADDR]], 5
; CHECK:   {{or|add}}
; CHECK:   %[[LOAD_SHADOW_PTR:[^ ]*]] = inttoptr
; CHECK:   %[[LOAD_SHADOW:[^ ]*]] = load i8, ptr %[[LOAD_SHADOW_PTR]]
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
; CHECK:   %tmp1 = load i32, ptr %a
; CHECK:   ret i32 %tmp1



entry:
  %tmp1 = load i32, ptr %a, align 4
  ret i32 %tmp1
}

define void @test_store(ptr %a) sanitize_address {
; CHECK-LABEL: @test_store
; CHECK-NOT: store
; CHECK:   %[[STORE_ADDR:[^ ]*]] = ptrtoint ptr %a to i64
; CHECK-S3:   lshr i64 %[[STORE_ADDR]], 3
; CHECK-S5:   lshr i64 %[[STORE_ADDR]], 5
; CHECK:   {{or|add}}
; CHECK:   %[[STORE_SHADOW_PTR:[^ ]*]] = inttoptr
; CHECK:   %[[STORE_SHADOW:[^ ]*]] = load i8, ptr %[[STORE_SHADOW_PTR]]
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
; CHECK:   store i32 42, ptr %a
; CHECK:   ret void
;

entry:
  store i32 42, ptr %a, align 4
  ret void
}

; Check that asan leaves just one alloca.

declare void @alloca_test_use(ptr)
define void @alloca_test() sanitize_address {
entry:
  %x = alloca [10 x i8], align 1
  %y = alloca [10 x i8], align 1
  %z = alloca [10 x i8], align 1
  call void @alloca_test_use(ptr %x)
  call void @alloca_test_use(ptr %y)
  call void @alloca_test_use(ptr %z)
  ret void
}

; CHECK-LABEL: define void @alloca_test()
; CHECK: %asan_local_stack_base = alloca
; CHECK: = alloca
; CHECK-NOT: = alloca
; CHECK: ret void

define void @LongDoubleTest(ptr nocapture %a) nounwind uwtable sanitize_address {
entry:
    store x86_fp80 0xK3FFF8000000000000000, ptr %a, align 16
    ret void
}

; CHECK-LABEL: LongDoubleTest
; CHECK: __asan_report_store_n
; CHECK: __asan_report_store_n
; CHECK: ret void


define void @i40test(ptr %a, ptr %b) nounwind uwtable sanitize_address {
  entry:
  %t = load i40, ptr %a
  store i40 %t, ptr %b, align 8
  ret void
}

; CHECK-LABEL: i40test
; CHECK: __asan_report_load_n{{.*}}, i64 5)
; CHECK: __asan_report_load_n{{.*}}, i64 5)
; CHECK: __asan_report_store_n{{.*}}, i64 5)
; CHECK: __asan_report_store_n{{.*}}, i64 5)
; CHECK: ret void

define void @i64test_align1(ptr %b) nounwind uwtable sanitize_address {
  entry:
  store i64 0, ptr %b, align 1
  ret void
}

; CHECK-LABEL: i64test_align1
; CHECK: __asan_report_store_n{{.*}}, i64 8)
; CHECK: __asan_report_store_n{{.*}}, i64 8)
; CHECK: ret void


define void @i80test(ptr %a, ptr %b) nounwind uwtable sanitize_address {
  entry:
  %t = load i80, ptr %a
  store i80 %t, ptr %b, align 8
  ret void
}

; CHECK-LABEL: i80test
; CHECK: __asan_report_load_n{{.*}}, i64 10)
; CHECK: __asan_report_load_n{{.*}}, i64 10)
; CHECK: __asan_report_store_n{{.*}}, i64 10)
; CHECK: __asan_report_store_n{{.*}}, i64 10)
; CHECK: ret void

; asan should not instrument functions with available_externally linkage.
define available_externally i32 @f_available_externally(ptr %a) sanitize_address  {
entry:
  %tmp1 = load i32, ptr %a
  ret i32 %tmp1
}
; CHECK-LABEL: @f_available_externally
; CHECK-NOT: __asan_report
; CHECK: ret i32


; CHECK-LABEL: @test_swifterror
; CHECK-NOT: __asan_report_load
; CHECK: ret void
define void @test_swifterror(ptr swifterror) sanitize_address {
  %swifterror_ptr_value = load ptr, ptr %0
  ret void
}

; CHECK-LABEL: @test_swifterror_2
; CHECK-NOT: __asan_report_store
; CHECK: ret void
define void @test_swifterror_2(ptr swifterror) sanitize_address {
  store ptr null, ptr %0
  ret void
}

; CHECK-LABEL: @test_swifterror_3
; CHECK-NOT: __asan_report_store
; CHECK: ret void
define void @test_swifterror_3() sanitize_address {
  %swifterror_addr = alloca swifterror ptr
  store ptr null, ptr %swifterror_addr
  call void @test_swifterror_2(ptr swifterror %swifterror_addr)
  ret void
}

;; ctor/dtor have the nounwind attribute. See uwtable.ll, they additionally have
;; the uwtable attribute with the module flag "uwtable".
; CHECK: define internal void @asan.module_ctor() #[[#ATTR:]] {{(comdat )?}}{
; CHECK: call void @__asan_init()

; CHECK: attributes #[[#ATTR]] = { nounwind }

; PROF
; CHECK: ![[PROF]] = !{!"branch_weights", i32 1, i32 100000}
