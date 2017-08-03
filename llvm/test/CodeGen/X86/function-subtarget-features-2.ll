; RUN: llc < %s -march=x86-64 -filetype=obj -o - | llvm-objdump -d - | FileCheck %s

; This test verifies that we assemble code for different architectures
; based on target-cpu and target-features attributes.
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo() #0 {
entry:
  call void asm sideeffect "aeskeygenassist  $$0x4, %xmm0, %xmm1", "~{dirflag},~{fpsr},~{flags}"()
  ret void
}

; CHECK: foo
; CHECK: aeskeygenassist

define void @bar() #2 {
entry:
  call void asm sideeffect "crc32b 4(%rbx), %eax", "~{dirflag},~{fpsr},~{flags}"()
  ret void
}

; CHECK: bar
; CHECK: crc32b

attributes #0 = { "target-cpu"="x86-64" "target-features"="+avx2" }
attributes #2 = { "target-cpu"="corei7" "target-features"="+sse4.2" }
