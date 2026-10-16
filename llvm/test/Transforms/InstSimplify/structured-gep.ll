; RUN: opt -S -passes=instsimplify < %s | FileCheck %s

%struct.S = type { i32, float }
%struct.Nested = type { %struct.S, i32 }

@g = external global i32
@arr = external global [10 x i32]

define ptr @zero_index_local(ptr %p) {
; CHECK-LABEL: @zero_index_local(ptr %p
; CHECK-NEXT:    ret ptr %p
;
  %sgep = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(i32) %p)
  ret ptr %sgep
}

define ptr @zero_index_global() {
; CHECK-LABEL: @zero_index_global(
; CHECK-NEXT:    ret ptr @g
;
  %sgep = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(i32) @g)
  ret ptr %sgep
}

define ptr @zero_index_array(ptr %p) {
; CHECK-LABEL: @zero_index_array(ptr %p
; CHECK-NEXT:    ret ptr %p
;
  %sgep = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([10 x i32]) %p)
  ret ptr %sgep
}

define ptr addrspace(11) @zero_index_addrspace(ptr addrspace(11) %p) {
; CHECK-LABEL: @zero_index_addrspace(ptr addrspace(11) %p
; CHECK-NEXT:    ret ptr addrspace(11) %p
;
  %sgep = call ptr addrspace(11) (ptr addrspace(11), ...) @llvm.structured.gep.p11(ptr addrspace(11) elementtype(i32) %p)
  ret ptr addrspace(11) %sgep
}

define i32 @zero_index_with_load(ptr %p) {
; CHECK-LABEL: @zero_index_with_load(ptr %p
; CHECK-NEXT:    [[V:%.*]] = load i32, ptr %p, align 4
; CHECK-NEXT:    ret i32 [[V]]
;
  %sgep = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(i32) %p)
  %v = load i32, ptr %sgep, align 4
  ret i32 %v
}

; GEP with a zero index can be simplified, but SGEP cannot.
define ptr @zero_index_struct(ptr %p) {
; CHECK-LABEL: @zero_index_struct(ptr %p
; CHECK-NEXT:    [[SGEP:%.*]] = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%struct.S) %p, i32 0)
; CHECK-NEXT:    ret ptr [[SGEP]]
;
  %sgep = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%struct.S) %p, i32 0)
  ret ptr %sgep
}

define ptr @one_index_struct(ptr %p) {
; CHECK-LABEL: @one_index_struct(ptr %p
; CHECK-NEXT:    [[SGEP:%.*]] = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%struct.S) %p, i32 1)
; CHECK-NEXT:    ret ptr [[SGEP]]
;
  %sgep = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%struct.S) %p, i32 1)
  ret ptr %sgep
}

define ptr @multi_index_nested(ptr %p) {
; CHECK-LABEL: @multi_index_nested(
; CHECK-NEXT:    [[SGEP:%.*]] = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%struct.Nested) %p, i32 0, i32 0)
; CHECK-NEXT:    ret ptr [[SGEP]]
;
  %sgep = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%struct.Nested) %p, i32 0, i32 0)
  ret ptr %sgep
}

define ptr @dynamic_index(ptr %p, i32 %i) {
; CHECK-LABEL: @dynamic_index(
; CHECK-NEXT:    [[SGEP:%.*]] = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([10 x i32]) %p, i32 %i)
; CHECK-NEXT:    ret ptr [[SGEP]]
;
  %sgep = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([10 x i32]) %p, i32 %i)
  ret ptr %sgep
}

declare ptr @llvm.structured.gep.p0(ptr, ...) #0
declare ptr addrspace(11) @llvm.structured.gep.p11(ptr addrspace(11), ...) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
