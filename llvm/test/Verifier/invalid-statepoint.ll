; RUN: not opt -passes=verify 2>&1 < %s | FileCheck %s

declare zeroext i1 @return0i1()

; Function Attrs: nounwind
declare token @llvm.experimental.gc.statepoint.p0f0i1f(i64, i32, i1 ()*, i32, i32, ...) #0

; Function Attrs: nounwind
declare i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token, i32, i32) #0

; CHECK: gc.statepoint callee argument must have elementtype attribute
define i32 addrspace(1)* @missing_elementtype(i32 addrspace(1)* %dparam) gc "statepoint-example" {
  %a00 = load i32, i32 addrspace(1)* %dparam
  %to0 = call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f0i1f(i64 0, i32 0, i1 ()* @return0i1, i32 9, i32 0, i2 0) ["gc-live" (i32 addrspace(1)* %dparam)]
  %relocate = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %to0, i32 0, i32 0)
  ret i32 addrspace(1)* %relocate
}

; CHECK: Attribute 'elementtype' type does not match parameter!
define i32 addrspace(1)* @elementtype_mismatch(i32 addrspace(1)* %dparam) gc "statepoint-example" {
  %a00 = load i32, i32 addrspace(1)* %dparam
  %to0 = call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f0i1f(i64 0, i32 0, i1 ()* elementtype(i32 ()) @return0i1, i32 9, i32 0, i2 0) ["gc-live" (i32 addrspace(1)* %dparam)]
  %relocate = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %to0, i32 0, i32 0)
  ret i32 addrspace(1)* %relocate
}

; CHECK: gc.statepoint mismatch in number of call args
define i32 addrspace(1)* @num_args_mismatch(i32 addrspace(1)* %dparam) gc "statepoint-example" {
  %a00 = load i32, i32 addrspace(1)* %dparam
  %to0 = call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f0i1f(i64 0, i32 0, i1 ()* elementtype(i1 ()) @return0i1, i32 9, i32 0, i2 0) ["gc-live" (i32 addrspace(1)* %dparam)]
  %relocate = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %to0, i32 0, i32 0)
  ret i32 addrspace(1)* %relocate
}

attributes #0 = { nounwind }
