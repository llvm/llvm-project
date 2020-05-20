; RUN: opt -attributor -S %s | FileCheck %s
; RUN: opt -passes=attributor -S %s | FileCheck %s

@var = internal global [1 x i32] undef

; CHECK-LABEL: define i32 addrspace(1)* @foo(i32 addrspace(4)* nofree readnone %arg)
define i32 addrspace(1)* @foo(i32 addrspace(4)* %arg) {
entry:
  %0 = addrspacecast i32 addrspace(4)* %arg to i32 addrspace(1)*
  ret i32 addrspace(1)* %0
}

define i32* @func1() {
  %ptr = call i32* @func1a([1 x i32]* @var)
  ret i32* %ptr
}

; CHECK-LABEL: define internal nonnull align 4 dereferenceable(4) i32* @func1a()
; CHECK-NEXT: ret i32* getelementptr inbounds ([1 x i32], [1 x i32]* @var, i32 0, i32 0)
define internal i32* @func1a([1 x i32]* %arg) {
  %ptr = getelementptr inbounds [1 x i32], [1 x i32]* %arg, i64 0, i64 0
  ret i32* %ptr
}
