; AIR has calls (no recursion): a large function called from many places
; stays a function, internal and called directly, where inlining it would
; add more than -air-keep-function-cost instructions. The system values it
; uses come from its callers, up to the kernels' parameters. A function
; called once, one using threadgroup memory and one taking a pointer are
; inlined; -air-keep-function-cost=0 inlines everything.
; RUN: llc -mtriple=air -air-keep-function-cost=1 -air-keep-function-min-size=2 %s -o - | FileCheck %s
; RUN: llc -mtriple=air -air-keep-function-cost=0 %s -o - | FileCheck %s --check-prefix=ALL
; RUN: llc -mtriple=air -filetype=obj -air-output=air -air-keep-function-cost=1 -air-keep-function-min-size=2 %s -o %t.air
; RUN: llvm-dis %t.air -o - | FileCheck %s --check-prefix=AIR

declare [3 x i32] @air.thread_position_in_grid()
declare i32 @air.thread_index_in_simdgroup()
declare i32 @air.simd_sum.s.i32(i32)

@tg = internal addrspace(3) global [4 x i32] undef

; CHECK-LABEL: define internal i32 @leaf(i32 %x, <3 x i32> %tid, i32 %simdlane)
; CHECK: %tid_x = extractelement <3 x i32> %tid, i32 0
; CHECK: mul i32 %x, %tid_x
; CHECK: add i32 %a, %simdlane
define i32 @leaf(i32 %x) {
  %t = call [3 x i32] @air.thread_position_in_grid()
  %tx = extractvalue [3 x i32] %t, 0
  %l = call i32 @air.thread_index_in_simdgroup()
  %a = mul i32 %x, %tx
  %b = add i32 %a, %l
  ret i32 %b
}

; CHECK-LABEL: define internal i32 @mid(i32 %x, <3 x i32> %tid, i32 %simdlane)
; CHECK: call i32 @leaf(i32 %x, <3 x i32> %tid, i32 %simdlane)
; CHECK: call i32 @leaf(i32 %a, <3 x i32> %tid, i32 %simdlane)
; (Its callers may call it under a branch.)
; CHECK: call void @llvm.sideeffect()
; CHECK-NEXT: call i32 @air.simd_sum.s.i32(
define i32 @mid(i32 %x) {
  %a = call i32 @leaf(i32 %x)
  %b = call i32 @leaf(i32 %a)
  %c = add i32 %a, %b
  %s = call i32 @air.simd_sum.s.i32(i32 %c)
  ret i32 %s
}

; CHECK-NOT: define {{.*}}@once
define i32 @once(i32 %x) {
  %a = mul i32 %x, 3
  %b = add i32 %a, 1
  ret i32 %b
}

; CHECK-NOT: define {{.*}}@shared
define i32 @shared(i32 %x) {
  %p = getelementptr [4 x i32], ptr addrspace(3) @tg, i32 0, i32 %x
  %v = load i32, ptr addrspace(3) %p
  %w = add i32 %v, %x
  ret i32 %w
}

; CHECK-NOT: define {{.*}}@pointer
define i32 @pointer(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p
  %w = add i32 %v, 1
  ret i32 %w
}

; CHECK-LABEL: define void @k1(ptr addrspace(1) {{.*}}%out, <3 x i32> %tid, i32 %simdlane)
; CHECK: call i32 @mid(i32 %tid_x, <3 x i32> %tid, i32 %simdlane)
; CHECK-NOT: call i32 @once
; CHECK-NOT: call i32 @shared
; CHECK-NOT: call i32 @pointer
define void @k1(ptr addrspace(1) %out) "air-kernel" {
  %t = call [3 x i32] @air.thread_position_in_grid()
  %tx = extractvalue [3 x i32] %t, 0
  %v = call i32 @mid(i32 %tx)
  %w = call i32 @once(i32 %v)
  %s = call i32 @shared(i32 %w)
  %u = call i32 @pointer(ptr addrspace(1) %out)
  %r = add i32 %s, %u
  %p = getelementptr i32, ptr addrspace(1) %out, i32 %tx
  store i32 %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: define void @k2(ptr addrspace(1) {{.*}}%out, <3 x i32> %tid, i32 %simdlane)
; CHECK: call i32 @mid(i32 7, <3 x i32> %tid, i32 %simdlane)
; CHECK: call i32 @mid(i32 %v, <3 x i32> %tid, i32 %simdlane)
; CHECK-NOT: call i32 @pointer
define void @k2(ptr addrspace(1) %out) "air-kernel" {
  %v = call i32 @mid(i32 7)
  %u = call i32 @mid(i32 %v)
  %q = call i32 @pointer(ptr addrspace(1) %out)
  %r = add i32 %u, %q
  store i32 %r, ptr addrspace(1) %out
  ret void
}

; CHECK: !air.kernel = !{[[K1:![0-9]+]], [[K2:![0-9]+]]}
; CHECK-DAG: [[K1]] = !{ptr @k1, !{{[0-9]+}}, [[ARGS:![0-9]+]]}
; CHECK-DAG: [[K2]] = !{ptr @k2, !{{[0-9]+}}, [[ARGS]]}
; CHECK-DAG: [[ARGS]] = !{!{{[0-9]+}}, [[TID:![0-9]+]], [[LANE:![0-9]+]]}
; CHECK-DAG: [[TID]] = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"tid"}
; CHECK-DAG: [[LANE]] = !{i32 2, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"simdlane"}

; ALL-NOT: define {{.*}}@leaf
; ALL-NOT: define {{.*}}@mid
; ALL: define void @k1(
; ALL-NOT: call i32 @{{[lmop]}}
; ALL: define void @k2(
; ALL-NOT: call i32 @{{[lmop]}}

; The functions in AIR's bitcode: internal, called with their types, the one
; with SIMD-group calls convergent.
; AIR: define internal i32 @leaf(i32 %0, <3 x i32> %1, i32 %2) {
; AIR: define internal i32 @mid(i32 %0, <3 x i32> %1, i32 %2) [[CONV:#[0-9]+]] {
; AIR: call i32 @leaf(i32 %0, <3 x i32> %1, i32 %2)
; AIR: define void @k1(
; AIR: call i32 @mid(
; AIR: define void @k2(
; AIR: call i32 @mid(i32 7,
; AIR: attributes [[CONV]] = { convergent }
