; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=+sse4.1 -O0 < %s | FileCheck %s

; Check that at -O0, the backend doesn't attempt to canonicalize a vector load
; used by an INSERTPS into a scalar load plus scalar_to_vector.
;
; In order to fold a load into the memory operand of an INSERTPSrm, the backend
; tries to canonicalize a vector load in input to an INSERTPS node into a
; scalar load plus scalar_to_vector. This would allow ISel to match the
; INSERTPSrm variant rather than a load plus INSERTPSrr.
;
; However, ISel can only select an INSERTPSrm if folding a load into the operand
; of an insertps is considered to be profitable.
;
; In the example below:
;
; __m128 test(__m128 a, __m128 *b) {
;   __m128 c = _mm_insert_ps(a, *b, 1 << 6);
;   return c;
; }
;
; At -O0, the backend would attempt to canonicalize the load to 'b' into
; a scalar load in the hope of matching an INSERTPSrm.
; However, ISel would fail to recognize an INSERTPSrm since load folding is
; always considered unprofitable at -O0. This would leave the insertps mask
; in an invalid state.
;
; The problem with the canonicalization rule performed by the backend is that
; it assumes ISel to always be able to match an INSERTPSrm. This assumption is
; not always correct at -O0. In this example, FastISel fails to lower the
; arguments needed by the entry block. This is enough to enable the DAGCombiner
; and eventually trigger the canonicalization on the INSERTPS node.
;
; This test checks that the vector load in input to the insertps is not
; canonicalized into a scalar load plus scalar_to_vector (a movss).

define <4 x float> @test(<4 x float> %a, ptr %b) {
; CHECK-LABEL: test:
; CHECK: movaps (%rdi), [[REG:%[a-z0-9]+]]
; CHECK-NOT: movss
; CHECK: insertps $64, [[REG]],
; CHECK: ret
entry:
  %0 = load <4 x float>, ptr %b, align 16
  %1 = call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %a, <4 x float> %0, i32 64)
  %2 = alloca <4 x float>, align 16
  store <4 x float> %1, ptr %2, align 16
  %3 = load <4 x float>, ptr %2, align 16
  ret <4 x float> %3
}


declare <4 x float> @llvm.x86.sse41.insertps(<4 x float>, <4 x float>, i32)
