; RUN: llvm-as < %s >%t1
; RUN: llvm-lto -exported-symbol=test_x86_aesni_aeskeygenassist -mattr=+aes -o %t2 %t1
; RUN: llvm-objdump -d %t2 | FileCheck -check-prefix=WITH_AES %s
; RUN: not --crash llvm-lto -exported-symbol=test_x86_aesni_aeskeygenassist -mattr=-aes -o %t3 %t1 2>&1 | FileCheck -check-prefix=WITHOUT_AES %s

target triple = "x86_64-unknown-linux-gnu"
declare <2 x i64> @llvm.x86.aesni.aeskeygenassist(<2 x i64>, i8)
define <2 x i64> @test_x86_aesni_aeskeygenassist(<2 x i64> %a0) {
  ; WITH_AES: test_x86_aesni_aeskeygenassist
  ; WITH_AES: aeskeygenassist
  %res = call <2 x i64> @llvm.x86.aesni.aeskeygenassist(<2 x i64> %a0, i8 7)
  ret <2 x i64> %res
}

; WITHOUT_AES: LLVM ERROR: Cannot select: intrinsic %llvm.x86.aesni.aeskeygenassist
