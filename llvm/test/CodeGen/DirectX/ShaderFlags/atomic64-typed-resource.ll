; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC

target triple = "dxil-pc-shadermodel6.6-compute"

; 64-bit atomic on a typed UAV (RWBuffer<uint64_t>) should set both
; Int64Ops (bit 20) and AtomicInt64OnTypedResource (bit 27), yielding 0x8100000.

; CHECK: ; Combined Shader Flags for Module
; CHECK-NEXT: ; Shader Flags Value: 0x08100000
; CHECK-NEXT: ;
; CHECK-NEXT: ; Note: shader requires additional functionality:
; CHECK-NEXT: ;       64-Bit integer
; CHECK-NEXT: ;       64-bit Atomics on Typed Resources
; CHECK-NEXT: ; Note: extra DXIL module flags:
; CHECK-NEXT: ;
; CHECK-NEXT: ; Shader Flags for Module Functions

; CHECK: Function main : 0x08100000
define void @main() #0 {
  %handle = call target("dx.TypedBuffer", i64, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i64_1_0_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)
  %old = call i64 @llvm.dx.resource.atomic.binop.i64(
      target("dx.TypedBuffer", i64, 1, 0, 0) %handle,
      i32 0, i32 0, i32 poison, i64 1)
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

; DXC: - Name:            SFI0
; DXC-NEXT:     Size:            8
; DXC-NEXT:     Flags:
; DXC:            Int64Ops:        true
; DXC:            AtomicInt64OnTypedResource: true
; DXC: ...
