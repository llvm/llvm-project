; RUN: opt -mtriple=hexagon -S -passes=dse < %s | FileCheck %s

; The HMX bias read instruction reads memory. Its intrinsic must not be marked
; as write-only, otherwise dead store elimination removes the store that fills
; the bias buffer, because the buffer is overwritten after the bias read.

; CHECK-LABEL: define void @bias_read(
; CHECK:       store i32 %v, ptr %bias
; CHECK-NEXT:  call void @llvm.hexagon.M8.mxmem2.bias(ptr %bias)
; CHECK-NEXT:  store i32 0, ptr %bias
define void @bias_read(ptr %bias, i32 %v) {
  store i32 %v, ptr %bias
  call void @llvm.hexagon.M8.mxmem2.bias(ptr %bias)
  store i32 0, ptr %bias
  ret void
}

; CHECK: declare void @llvm.hexagon.M8.mxmem2.bias(ptr) [[ATTRS:#[0-9]+]]
; CHECK-NOT: attributes [[ATTRS]] = {{.*}}memory(
declare void @llvm.hexagon.M8.mxmem2.bias(ptr)
