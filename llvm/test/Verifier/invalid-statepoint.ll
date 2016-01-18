; RUN: not opt -verify 2>&1 < %s | FileCheck %s

; CHECK: gc.statepoint: mismatch in number of call arguments 

declare zeroext i1 @return0i1()

; Function Attrs: nounwind
declare i32 @llvm.experimental.gc.statepoint.p0f0i1f(i64, i32, i1 ()*, i32, i32, ...) #0

; Function Attrs: nounwind
declare i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(i32, i32, i32) #0

define i32 addrspace(1)* @0(i32 addrspace(1)* %dparam) {
  %a00 = load i32, i32 addrspace(1)* %dparam
  %to0 = call i32 (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f0i1f(i64 0, i32 0, i1 ()* @return0i1, i32 9, i32 0, i2 0, i32 addrspace(1)* %dparam)
  %relocate = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(i32 %to0, i32 2, i32 6)
  ret i32 addrspace(1)* %relocate
}

attributes #0 = { nounwind }