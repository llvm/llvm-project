; RUN: llc --filetype=obj %s --stop-after=dxil-write-bitcode -o %t.bc
; RUN: dxil-dis %t.bc -o - | FileCheck %s

; Verify that the DXIL reader supports the atomicrmw record.

target triple = "dxil-pc-shadermodel6.0-compute"

@gsm = internal addrspace(3) global i32 zeroinitializer, align 4

; CHECK-LABEL: define void @main()
; CHECK: atomicrmw add i32 addrspace(3)* @gsm, i32 1 monotonic

define void @main() #0 {
  %old = atomicrmw add ptr addrspace(3) @gsm, i32 1 monotonic
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
