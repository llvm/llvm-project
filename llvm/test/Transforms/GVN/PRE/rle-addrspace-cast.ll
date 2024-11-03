; RUN: opt < %s -data-layout="e-p:32:32:32-p1:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-n8:16:32" -passes=gvn,dce -S | FileCheck %s

define i8 @coerce_offset0_addrspacecast(i32 %V, ptr %P) {
  store i32 %V, ptr %P

  %P2 = addrspacecast ptr %P to ptr addrspace(1)
  %P3 = getelementptr i8, ptr addrspace(1) %P2, i32 2

  %A = load i8, ptr addrspace(1) %P3
  ret i8 %A
; CHECK-LABEL: @coerce_offset0_addrspacecast(
; CHECK-NOT: load
; CHECK: ret i8
}
