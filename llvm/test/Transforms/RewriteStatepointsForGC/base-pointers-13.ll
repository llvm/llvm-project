; RUN: opt < %s -rewrite-statepoints-for-gc -spp-print-base-pointers -S 2>&1 | FileCheck %s
; RUN: opt < %s -passes=rewrite-statepoints-for-gc -spp-print-base-pointers -S 2>&1 | FileCheck %s

; CHECK: derived %derived base null

@global = external addrspace(1) global i8

define i8 @test(i64 %offset) gc "statepoint-example" {
  %derived = getelementptr i8, i8 addrspace(1)* @global, i64 %offset
  call void @extern()
; CHECK-NOT: relocate
; CHECK-NOT: remat
; CHECK: %load = load i8, i8 addrspace(1)* %derived
  %load = load i8, i8 addrspace(1)* %derived
  ret i8 %load
}

declare void @extern() gc "statepoint-example"

declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)
