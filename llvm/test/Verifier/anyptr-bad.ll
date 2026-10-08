; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s

; CHECK: intrinsic return type (overload type 0) expected any pointer type, but got <2 x ptr>
; CHECK-NEXT: declare <2 x ptr> @llvm.returnaddress(i32)
declare <2 x ptr> @llvm.returnaddress(i32)
