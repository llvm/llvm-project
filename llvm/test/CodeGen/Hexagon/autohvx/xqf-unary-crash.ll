; Tests that unary qf instruction handling in postRA xqf handler
; does not cause a crash
; REQUIRES: asserts

; RUN: llc -mhvx -mcpu=hexagonv81 -mattr=+hvxv81,+hvx-length128b,+hvx-qfloat \
; RUN: -enable-xqf-gen=true -hexagon-qfloat-mode=ieee %s -o /dev/null

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

declare i32 @printf(...)

define i32 @main(ptr %0) {
entry:
  store <32 x i32> zeroinitializer, ptr %0, align 128
  %call3 = call i32 (...) @printf()
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vilog2.qf16.128B(<32 x i32> zeroinitializer)
  store <32 x i32> %1, ptr %0, align 128
  ret i32 0
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vilog2.qf16.128B(<32 x i32>) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(none) }
