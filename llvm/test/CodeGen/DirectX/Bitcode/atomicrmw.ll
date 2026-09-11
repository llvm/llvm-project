; RUN: llc --filetype=obj %s --stop-after=dxil-write-bitcode -o %t.bc
; RUN: llvm-bcanalyzer --dump --non-symbolic --disable-histogram %t.bc | \
; RUN:   FileCheck %s --check-prefix=BITCODE
; RUN: llvm-dis %t.bc -o - | FileCheck %s --check-prefix=LLVM

; Verify that DXIL uses atomicrmw record code 38, which the LLVM reader
; supports.

target triple = "dxil-pc-shadermodel6.0-compute"

@gsm = internal addrspace(3) global i32 zeroinitializer, align 4

; LLVM-LABEL: define void @main()
; LLVM: atomicrmw add ptr addrspace(3) @gsm, i32 1 monotonic
; BITCODE-LABEL: <FUNCTION_BLOCK
; BITCODE: <INST_ATOMICRMW_OLD codeid=38
; BITCODE-NEXT: <INST_RET

define void @main() #0 {
  %old = atomicrmw add ptr addrspace(3) @gsm, i32 1 monotonic
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
