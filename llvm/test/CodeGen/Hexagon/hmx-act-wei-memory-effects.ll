; RUN: opt -mtriple=hexagon -S -passes=dse < %s | FileCheck %s

; HMX activation and weight instructions read memory. Their intrinsics must
; not be marked as write-only, otherwise dead store elimination removes the
; stores that fill the activation and weight buffers, because the buffers are
; overwritten after the HMX instructions.

; CHECK-LABEL: define void @act_wei_pair(
; CHECK:       store i32 %v0, ptr %act
; CHECK-NEXT:  store i32 %v1, ptr %wei
; CHECK-NEXT:  ptrtoint
; CHECK-NEXT:  ptrtoint
; CHECK-NEXT:  call void @llvm.hexagon.M8.mxmem.blk.sm.act.hf(
; CHECK-NEXT:  call void @llvm.hexagon.M8.mxmem.wei.hf(
; CHECK-NEXT:  store i32 0, ptr %act
; CHECK-NEXT:  store i32 0, ptr %wei
define void @act_wei_pair(ptr %act, ptr %wei, i32 %v0, i32 %v1, i32 %n) {
  store i32 %v0, ptr %act
  store i32 %v1, ptr %wei
  %a = ptrtoint ptr %act to i32
  %w = ptrtoint ptr %wei to i32
  call void @llvm.hexagon.M8.mxmem.blk.sm.act.hf(i32 %a, i32 %n)
  call void @llvm.hexagon.M8.mxmem.wei.hf(i32 %w, i32 %n)
  store i32 0, ptr %act
  store i32 0, ptr %wei
  ret void
}

; CHECK: declare void @llvm.hexagon.M8.mxmem.blk.sm.act.hf(i32, i32) [[ATTRS:#[0-9]+]]
; CHECK: declare void @llvm.hexagon.M8.mxmem.wei.hf(i32, i32) [[ATTRS]]
; CHECK-NOT: attributes [[ATTRS]] = {{.*}}memory(
declare void @llvm.hexagon.M8.mxmem.blk.sm.act.hf(i32, i32)
declare void @llvm.hexagon.M8.mxmem.wei.hf(i32, i32)
