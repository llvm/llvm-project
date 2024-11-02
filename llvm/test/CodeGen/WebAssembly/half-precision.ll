; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+half-precision | FileCheck %s
; RUN: llc < %s --mtriple=wasm64-unknown-unknown -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+half-precision | FileCheck %s

declare float @llvm.wasm.loadf32.f16(ptr)
declare void @llvm.wasm.storef16.f32(float, ptr)

; CHECK-LABEL: ldf16_32:
; CHECK:      f32.load_f16 $push[[NUM0:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM0]]{{$}}
define float @ldf16_32(ptr %p) {
  %v = call float @llvm.wasm.loadf16.f32(ptr %p)
  ret float %v
}

; CHECK-LABEL: stf16_32:
; CHECK:       f32.store_f16 0($1), $0
; CHECK-NEXT:  return
define void @stf16_32(float %v, ptr %p) {
  tail call void @llvm.wasm.storef16.f32(float %v, ptr %p)
  ret void
}
