; RUN: opt -S -dxil-resource-access -mtriple=dxil -verify-each %s | FileCheck %s
;
; GEP indices may be wider than the pointer index type. Truncate to the
; 32-bit address-space index type before scaling a two-row array element.

target datalayout = "e-p:32:32-p2:32:32-i64:64-n8:16:32:64"

%__cblayout_CB = type { [4 x <8 x i32>] }

; CHECK-LABEL: define i32 @load_two_row_element(
; CHECK: [[INDEX:%.*]] = trunc i64 %idx to i32
; CHECK: [[ROW:%.*]] = mul i32 [[INDEX]], 2
; CHECK: [[LOAD:%.*]] = call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", %__cblayout_CB) {{%.*}}, i32 [[ROW]])
; CHECK: [[VALUE:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 0
; CHECK: ret i32 [[VALUE]]
define i32 @load_two_row_element(i64 %idx) {
entry:
  %handle = call target("dx.CBuffer", %__cblayout_CB) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %base = call ptr addrspace(2) @llvm.dx.resource.getpointer(target("dx.CBuffer", %__cblayout_CB) %handle, i32 0)
  %element = getelementptr <8 x i32>, ptr addrspace(2) %base, i64 %idx
  %value = load i32, ptr addrspace(2) %element, align 16
  ret i32 %value
}
