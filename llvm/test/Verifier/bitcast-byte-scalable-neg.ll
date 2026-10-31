; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s
;
; With p:64:
;   <2 x ptr>          = 2*64 = 128 bits (fixed)
;   <vscale x 4 x b32> = 4*32 = 128*vscale bits (scalable)
;   <vscale x 4 x ptr> = 4*64 = 256*vscale bits (scalable)
;
; A bitcast requires identical sizes, so neither pair is bitcastable: %a has
; the same minimum size but different scalability, %b has different minimum
; sizes.

target datalayout = "e-p:64:64:64"

; CHECK: Invalid bitcast
; CHECK: %a = bitcast <2 x ptr> %vp to <vscale x 4 x b32>
; CHECK: Invalid bitcast
; CHECK: %b = bitcast <vscale x 4 x ptr> %sp to <vscale x 4 x b32>
define void @f(<2 x ptr> %vp, <vscale x 4 x ptr> %sp) {
  %a = bitcast <2 x ptr> %vp to <vscale x 4 x b32>
  %b = bitcast <vscale x 4 x ptr> %sp to <vscale x 4 x b32>
  ret void
}
