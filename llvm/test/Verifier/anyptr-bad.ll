; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s

; CHECK: intrinsic return type (overload type 0) expected any pointer type, but got <2 x ptr>
; CHECK-NEXT: ptr @llvm.returnaddress

declare <2 x ptr> @llvm.returnaddress(i32)

define void @foo() {
  call <2 x ptr> @llvm.returnaddress(i32 0)
  ret void
}
