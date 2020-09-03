; RUN: not --crash llc < %s -mtriple=i686-unknown-unknown -mattr=sse2 2>&1 | FileCheck %s --check-prefix=CHECK

; Make sure we generate fatal error from the type legalizer for using a 64-bit
; mode intrinsics in 32-bit mode. We used to use an llvm_unreachable.

; CHECK: LLVM ERROR: Do not know how to expand the result of this operator!
define i64 @test_x86_sse2_cvtsd2si64(<2 x double> %a0) {
  %res = call i64 @llvm.x86.sse2.cvtsd2si64(<2 x double> %a0) ; <i64> [#uses=1]
  ret i64 %res
}
declare i64 @llvm.x86.sse2.cvtsd2si64(<2 x double>) nounwind readnone
