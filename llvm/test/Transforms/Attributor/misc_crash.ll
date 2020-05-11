; RUN: opt -attributor -S %s | FileCheck %s
; RUN: opt -passes=attributor -S %s | FileCheck %s

@var1 = internal global [1 x i32] undef
@var2 = internal global i32 0

; CHECK-LABEL: define i32 addrspace(1)* @foo(i32 addrspace(4)* nofree readnone %arg)
define i32 addrspace(1)* @foo(i32 addrspace(4)* %arg) {
entry:
  %0 = addrspacecast i32 addrspace(4)* %arg to i32 addrspace(1)*
  ret i32 addrspace(1)* %0
}

define i32* @func1() {
  %ptr = call i32* @func1a([1 x i32]* @var1)
  ret i32* %ptr
}

; CHECK-LABEL: define internal nonnull align 4 dereferenceable(4) i32* @func1a()
; CHECK-NEXT: ret i32* getelementptr inbounds ([1 x i32], [1 x i32]* @var1, i32 0, i32 0)
define internal i32* @func1a([1 x i32]* %arg) {
  %ptr = getelementptr inbounds [1 x i32], [1 x i32]* %arg, i64 0, i64 0
  ret i32* %ptr
}

define internal void @func2a(i32* %0) {
  store i32 0, i32* %0
  ret void
}

; CHECK-LABEL: define i32 @func2()
; CHECK-NEXT:   tail call void @func2a()
; CHECK-NEXT:   %1 = load i32, i32* @var2, align 4
; CHECK-NEXT:   ret i32 %1
define i32 @func2() {
  %1 = tail call i32 (i32*, ...) bitcast (void (i32*)* @func2a to i32 (i32*, ...)*)(i32* @var2)
  %2 = load i32, i32* @var2
  ret i32 %2
}
