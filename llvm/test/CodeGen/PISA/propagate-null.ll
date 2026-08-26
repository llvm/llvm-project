; RUN: llc < %s -march=pisa -stop-after=pisa-propagate-null-pointers | FileCheck %s

; CHECK: @global_null_ptr = dso_local local_unnamed_addr addrspace(1) global ptr addrspace(3) inttoptr (i32 -1 to ptr addrspace(3)), align 4
; CHECK-DAG: @shared_null = {{.*}} addrspace(3) inttoptr (i32 -1 to ptr addrspace(3))
; CHECK-DAG: @private_null = {{.*}} addrspace(4) inttoptr (i32 -1 to ptr addrspace(4))
@global_null_ptr = dso_local local_unnamed_addr addrspace(1) global ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)), align 4
@global_shared = addrspace(3) global i32 0, align 4

; CHECK-LABEL: @test_generic_to_shared
define ptr addrspace(3) @test_generic_to_shared(ptr %arg) {
  ; CHECK: [[CAST:%.*]] = addrspacecast ptr %arg to ptr addrspace(3)
  ; CHECK-NEXT: [[PRED:%.*]] = icmp ne ptr %arg, null
  ; CHECK-NEXT: [[SEL:%.*]] = select i1 [[PRED]], ptr addrspace(3) [[CAST]], ptr addrspace(3) inttoptr (i32 -1 to ptr addrspace(3))
  ; CHECK-NEXT: ret ptr addrspace(3) [[SEL]]
  %ptr = addrspacecast ptr %arg to ptr addrspace(3)
  ret ptr addrspace(3) %ptr
}

; CHECK-LABEL: @test_shared_to_generic
define ptr @test_shared_to_generic(ptr addrspace(3) %arg) {
  ; CHECK: [[CAST:%.*]] = addrspacecast ptr addrspace(3) %arg to ptr
  ; CHECK-NEXT: [[PRED:%.*]] = icmp ne ptr addrspace(3) %arg, inttoptr (i32 -1 to ptr addrspace(3))
  ; CHECK-NEXT: [[SEL:%.*]] = select i1 [[PRED]], ptr [[CAST]], ptr null
  ; CHECK-NEXT: ret ptr [[SEL]]
  %ptr = addrspacecast ptr addrspace(3) %arg to ptr
  ret ptr %ptr
}

; CHECK-LABEL: @test_generic_to_private
define ptr addrspace(4) @test_generic_to_private(ptr %arg) {
  ; CHECK: [[CAST:%.*]] = addrspacecast ptr %arg to ptr
  ; CHECK-NEXT: [[PRED:%.*]] = icmp ne ptr %arg, null
  ; CHECK-NEXT: [[SEL:%.*]] = select i1 [[PRED]], ptr addrspace(4) [[CAST]], ptr addrspace(4) inttoptr (i32 -1 to ptr addrspace(4))
  ; CHECK-NEXT: ret ptr addrspace(4) [[SEL]]
  %ptr = addrspacecast ptr %arg to ptr addrspace(4)
  ret ptr addrspace(4) %ptr
}

; CHECK-LABEL: @test_private_to_generic
define ptr @test_private_to_generic(ptr addrspace(4) %arg) {
  ; CHECK: [[CAST:%.*]] = addrspacecast ptr addrspace(4) %arg to ptr
  ; CHECK-NEXT: [[PRED:%.*]] = icmp ne ptr addrspace(4) %arg, inttoptr (i32 -1 to ptr addrspace(4))
  ; CHECK-NEXT: [[SEL:%.*]] = select i1 [[PRED]], ptr [[CAST]], ptr null
  ; CHECK-NEXT: ret ptr [[SEL]]
  %ptr = addrspacecast ptr addrspace(4) %arg to ptr
  ret ptr %ptr
}

; CHECK-LABEL: @test_generic_to_global
define ptr addrspace(1) @test_generic_to_global(ptr %arg) {
  ; CHECK: %ptr = addrspacecast ptr %arg to ptr addrspace(1)
  ; CHECK-NEXT: ret ptr addrspace(1) %ptr
  %ptr = addrspacecast ptr %arg to ptr addrspace(1)
  ret ptr addrspace(1) %ptr
}

; CHECK-LABEL: @test_global_to_generic
define ptr @test_global_to_generic(ptr addrspace(1) %arg) {
  ; CHECK: %ptr = addrspacecast ptr addrspace(1) %arg to ptr
  ; CHECK-NEXT: ret ptr %ptr
  %ptr = addrspacecast ptr addrspace(1) %arg to ptr
  ret ptr %ptr
}

; CHECK-LABEL: @test_alloca_private
define ptr @test_alloca_private() {
  ; CHECK: %var = alloca i32, align 4, addrspace(4)
  ; CHECK-NEXT: %ptr = addrspacecast ptr addrspace(4) %var to ptr
  ; CHECK-NEXT: ret ptr %ptr
  %var = alloca i32, align 4, addrspace(4)
  %ptr = addrspacecast ptr addrspace(4) %var to ptr
  ret ptr %ptr
}

; CHECK-LABEL: @test_global_in_shared
define ptr @test_global_in_shared() {
  ; CHECK: %ptr = addrspacecast ptr addrspace(3) @global_shared to ptr
  ; CHECK-NEXT: ret ptr %ptr
  %ptr = addrspacecast ptr addrspace(3) @global_shared to ptr
  ret ptr %ptr
}

; CHECK-LABEL: @test_nonnull_arg
define ptr @test_nonnull_arg(ptr addrspace(4) noundef nonnull %arg) {
  ; CHECK: %ptr = addrspacecast ptr addrspace(4) %arg to ptr
  ; CHECK-NEXT: ret ptr %ptr
  %ptr = addrspacecast ptr addrspace(4) %arg to ptr
  ret ptr %ptr
}

; CHECK-LABEL: @test_kernel_arg_shared
define pisa_kernel void @test_kernel_arg_shared(ptr addrspace(3) %arg, ptr addrspace(1) %res) {
  ; CHECK: %ptr = addrspacecast ptr addrspace(3) %arg to ptr
  ; CHECK-NEXT: store ptr %ptr, ptr addrspace(1) %res, align 4
  ; CHECK-NEXT: ret void
  %ptr = addrspacecast ptr addrspace(3) %arg to ptr
  store ptr %ptr, ptr addrspace(1) %res, align 4
  ret void
}

%struct.byval = type { i32, i16, i64, i8 }

; CHECK-LABEL: @test_kernel_arg_private
define pisa_kernel void @test_kernel_arg_private(ptr addrspace(4) byval(%struct.byval) %arg, ptr addrspace(1) %res) {
  ; CHECK: %ptr = addrspacecast ptr addrspace(4) %arg to ptr
  ; CHECK-NEXT: store ptr %ptr, ptr addrspace(1) %res, align 4
  ; CHECK-NEXT: ret void
  %ptr = addrspacecast ptr addrspace(4) %arg to ptr
  store ptr %ptr, ptr addrspace(1) %res, align 4
  ret void
}

; CHECK-LABEL: @test_kernel_arg_different_memory
define pisa_kernel void @test_kernel_arg_different_memory(ptr addrspace(1) %arg, ptr addrspace(1) %res) {
  ; CHECK: [[CAST:%.*]] = addrspacecast ptr %cast to ptr addrspace(3)
  ; CHECK-NEXT: [[PRED:%.*]] = icmp ne ptr %cast, null
  ; CHECK-NEXT: [[RES:%.*]] = select i1 [[PRED]], ptr addrspace(3) [[CAST]], ptr addrspace(3) inttoptr (i32 -1 to ptr addrspace(3))
  ; CHECK-NEXT: store ptr addrspace(3) [[RES]], ptr addrspace(1) %res, align 4
  %cast = addrspacecast ptr addrspace(1) %arg to ptr
  %cast_2 = addrspacecast ptr %cast to ptr addrspace(3)
  store ptr addrspace(3) %cast_2, ptr addrspace(1) %res, align 4
  ret void
}

; CHECK-LABEL: @test_constexpr_shared_null
define i1 @test_constexpr_shared_null(i32 %val) {
  ; CHECK: %cmp = icmp ne i32 %val, ptrtoint (ptr addrspace(3) inttoptr (i32 -1 to ptr addrspace(3)) to i32)
  ; CHECK-NEXT: ret i1 %cmp
  %cmp = icmp ne i32 %val, ptrtoint (ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)) to i32)
  ret i1 %cmp
}

; CHECK-LABEL: @test_constexpr_private_null
define i1 @test_constexpr_private_null(i32 %val) {
  ; CHECK: %cmp = icmp ne i32 %val, ptrtoint (ptr addrspace(4) inttoptr (i32 -1 to ptr addrspace(4)) to i32)
  ; CHECK-NEXT: ret i1 %cmp
  %cmp = icmp ne i32 %val, ptrtoint (ptr addrspace(4) addrspacecast (ptr null to ptr addrspace(4)) to i32)
  ret i1 %cmp
}

; CHECK-LABEL: @test_complex_expression
define ptr @test_complex_expression(i1 %cond) {
  ; CHECK: [[CAST:%.*]] = addrspacecast ptr addrspace(3) %val to ptr
  ; CHECK-NEXT: [[PRED:%.*]] = icmp ne ptr addrspace(3) %val, inttoptr (i32 -1 to ptr addrspace(3))
  ; CHECK-NEXT: [[SEL:%.*]] = select i1 [[PRED]], ptr [[CAST]], ptr null
  %val = select i1 %cond, ptr addrspace(3) @global_shared, ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3))
  %res = addrspacecast ptr addrspace(3) %val to ptr
  ret ptr %res
}

; Both globals use constant expression casts from generic null to private/shared.
; This ensures both GenericToPrivateCast and GenericToSharedCast are non-empty,
; so the pass marks the module as changed and replaces const expr casts.

@shared_null = addrspace(1) global ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)), align 4

@private_null = addrspace(1) global ptr addrspace(4) addrspacecast (ptr null to ptr addrspace(4)), align 4

; A function that compares against both shared and private null constant exprs,
; keeping both uses alive so the pass must replace both addrspacecasts.
; CHECK-LABEL: @test_both_constexpr_used
define i1 @test_both_constexpr_used(i32 %val, i32 %val2) {
  ; CHECK: %cmp_shared = icmp ne i32 %val, ptrtoint (ptr addrspace(3) inttoptr (i32 -1 to ptr addrspace(3)) to i32)
  %cmp_shared = icmp ne i32 %val, ptrtoint (ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)) to i32)

  ; CHECK-NEXT: %cmp_private = icmp ne i32 %val2, ptrtoint (ptr addrspace(4) inttoptr (i32 -1 to ptr addrspace(4)) to i32)
  %cmp_private = icmp ne i32 %val2, ptrtoint (ptr addrspace(4) addrspacecast (ptr null to ptr addrspace(4)) to i32)

  ; CHECK-NEXT: %result = and i1 %cmp_shared, %cmp_private
  %result = and i1 %cmp_shared, %cmp_private
  ; CHECK-NEXT: ret i1 %result
  ret i1 %result
}
