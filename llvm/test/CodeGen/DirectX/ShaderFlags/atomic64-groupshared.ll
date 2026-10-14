; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC

target triple = "dxil-pc-shadermodel6.6-compute"

; 64-bit atomic on groupshared memory (address space 3) should set both
; Int64Ops (bit 20) and AtomicInt64OnGroupShared (bit 28), yielding 0x10100000.

@gsm = internal addrspace(3) global i64 zeroinitializer, align 8

; CHECK: ; Combined Shader Flags for Module
; CHECK-NEXT: ; Shader Flags Value: 0x10100000
; CHECK-NEXT: ;
; CHECK-NEXT: ; Note: shader requires additional functionality:
; CHECK-NEXT: ;       64-Bit integer
; CHECK-NEXT: ;       64-bit Atomics on Group Shared
; CHECK-NEXT: ; Note: extra DXIL module flags:
; CHECK-NEXT: ;
; CHECK-NEXT: ; Shader Flags for Module Functions

; CHECK: Function main : 0x10100000
define void @main() #0 {
  %old = atomicrmw add ptr addrspace(3) @gsm, i64 1 monotonic
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

; DXC: - Name:            SFI0
; DXC-NEXT:     Size:            8
; DXC-NEXT:     Flags:
; DXC:            Int64Ops:        true
; DXC:            AtomicInt64OnGroupShared: true
; DXC: ...
