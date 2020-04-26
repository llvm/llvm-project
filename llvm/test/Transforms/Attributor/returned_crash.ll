; RUN: opt -attributor -S %s | FileCheck %s
; RUN: opt -passes=attributor -S %s | FileCheck %s
;
; CHECK: define i32 addrspace(1)* @foo(i32 addrspace(4)* nofree readnone %arg)
define i32 addrspace(1)* @foo(i32 addrspace(4)* %arg) {
entry:
  %0 = addrspacecast i32 addrspace(4)* %arg to i32 addrspace(1)*
  ret i32 addrspace(1)* %0
}
