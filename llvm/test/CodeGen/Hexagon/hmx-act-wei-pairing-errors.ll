; RUN: llc -O0 -verify-machineinstrs -mtriple=hexagon -mcpu=hexagonv68 \
; RUN:   -mattr=+hmx < %s -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --implicit-check-not=error:

; HMX activation and weight instructions must be in the same packet. Check that
; a warning is reported when a pair cannot be formed, because neither
; instruction can be moved past the instructions between them.

@abuf = external global [64 x i32]
@wbuf = external global [64 x i32]
@other = external global i32

; The addresses are plain integers, so the store may alias both buffers.
; CHECK: warning: In may_alias, expected activate and weight pair
define void @may_alias(i32 %a, i32 %w, i32 %n) {
  call void @llvm.hexagon.M8.mxmem.blk.sm.act.hf(i32 %a, i32 %n)
  store i32 1, ptr @other
  call void @llvm.hexagon.M8.mxmem.wei.hf(i32 %w, i32 %n)
  ret void
}

; A volatile store must not be reordered with the HMX instructions.
; CHECK: warning: In volatile_store, expected activate and weight pair
define void @volatile_store(i32 %n) {
  %a = ptrtoint ptr @abuf to i32
  %w = ptrtoint ptr @wbuf to i32
  call void @llvm.hexagon.M8.mxmem.blk.sm.act.hf(i32 %a, i32 %n)
  store volatile i32 1, ptr @other
  call void @llvm.hexagon.M8.mxmem.wei.hf(i32 %w, i32 %n)
  ret void
}

; Another HMX instruction between the activation and the weight.
; CHECK: warning: In hmx_between, expected activate and weight pair
define void @hmx_between(i32 %n) {
  %a = ptrtoint ptr @abuf to i32
  %w = ptrtoint ptr @wbuf to i32
  call void @llvm.hexagon.M8.mxmem.blk.sm.act.hf(i32 %a, i32 %n)
  call void @llvm.hexagon.M8.mxclracc.hf()
  call void @llvm.hexagon.M8.mxmem.wei.hf(i32 %w, i32 %n)
  ret void
}

; Alias analysis is not used, so the instructions are not moved past a store,
; even if it does not alias the memory read by either of them.
; CHECK: warning: In sink, expected activate and weight pair
define void @sink(i32 %n) {
  %a = ptrtoint ptr @abuf to i32
  %w = ptrtoint ptr @wbuf to i32
  call void @llvm.hexagon.M8.mxmem.blk.sm.act.hf(i32 %a, i32 %n)
  store i32 1, ptr @other
  call void @llvm.hexagon.M8.mxmem.wei.hf(i32 %w, i32 %n)
  ret void
}

; The store writes the activation buffer, so the activation must stay before
; it. The weight is not moved up past the store either, although the store does
; not alias the weight buffer.
; CHECK: warning: In hoist, expected activate and weight pair
define void @hoist(i32 %n) {
  %a = ptrtoint ptr @abuf to i32
  %w = ptrtoint ptr @wbuf to i32
  call void @llvm.hexagon.M8.mxmem.blk.sm.act.hf(i32 %a, i32 %n)
  store i32 1, ptr @abuf
  call void @llvm.hexagon.M8.mxmem.wei.hf(i32 %w, i32 %n)
  ret void
}

; A call between the activation and the weight.
; CHECK: warning: In call_between, expected activate and weight pair
define void @call_between(i32 %n) {
  %a = ptrtoint ptr @abuf to i32
  %w = ptrtoint ptr @wbuf to i32
  call void @llvm.hexagon.M8.mxmem.blk.sm.act.hf(i32 %a, i32 %n)
  call void @g()
  call void @llvm.hexagon.M8.mxmem.wei.hf(i32 %w, i32 %n)
  ret void
}

declare void @g()
declare void @llvm.hexagon.M8.mxclracc.hf()
declare void @llvm.hexagon.M8.mxmem.blk.sm.act.hf(i32, i32)
declare void @llvm.hexagon.M8.mxmem.wei.hf(i32, i32)
