; RUN: llc --filetype=obj %s --stop-after=dxil-write-bitcode -o %t.bc
; RUN: llvm-bcanalyzer --dump --non-symbolic --disable-histogram %t.bc | \
; RUN:   FileCheck %s --check-prefix=BITCODE
; RUN: dxil-dis %t.bc -o - | FileCheck %s --check-prefix=DXIL
; RUN: llvm-dis %t.bc -o - | FileCheck %s --check-prefix=LLVM

; Verify that DXIL uses atomicrmw record code 38, which both the DXIL and LLVM
; readers support.

target triple = "dxil-pc-shadermodel6.0-compute"

@gsm = internal addrspace(3) global i32 zeroinitializer, align 4

; DXIL-LABEL: define void @main()
; DXIL: atomicrmw add i32 addrspace(3)* @gsm, i32 1 monotonic
; LLVM-LABEL: define void @main()
; LLVM: atomicrmw add ptr addrspace(3) @gsm, i32 1 monotonic
; BITCODE-LABEL: <FUNCTION_BLOCK
; BITCODE: <UnknownCode38
; BITCODE-NEXT: <INST_RET

define void @main() #0 {
  %old = atomicrmw add ptr addrspace(3) @gsm, i32 1 monotonic
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
