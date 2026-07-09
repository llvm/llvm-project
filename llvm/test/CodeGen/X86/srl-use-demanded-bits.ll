; RUN: llc -mtriple=x86_64-unknown-linux < %s | FileCheck %s
;
; visitSRL computes the actual demanded bits of the SRL result from downstream
; users before calling SimplifyDemandedBits.  When the upstream operand is an
; AND whose mask only clears bits that no downstream user ever reads, the AND
; is redundant and should be eliminated.

; Simple case: and(x, ~255) >> 8 & 255  ==  x >> 8 & 255
; The AND with ~255 zeroes the low byte, but after shifting right by 8 those
; bits shift out of frame and are never read by the downstream AND with 255.
; The upstream AND must not appear in the output.
define i64 @demand_narrows_upstream_and(i64 %x) {
; CHECK-LABEL: demand_narrows_upstream_and:
; CHECK-NOT:   andq
; CHECK:       movzbl %ah, %eax
; CHECK-NEXT:  retq
  %a = and i64 %x, -256
  %s = lshr i64 %a, 8
  %r = and i64 %s, 255
  ret i64 %r
}

; Multi-hop case: demand flows through two XOR / SRL hops back to an upstream
; AND.  The AND clears the low byte; the final AND with 15 only reads bits[0:3]
; of the chain.  Tracing backward: bits[0:3] demanded from xor2, bits[0:7]
; demanded from xor1 and from srl(%a,8), bits[8:15] demanded from %a.  The
; upstream AND clears bits[0:7] which are outside bits[8:15], so it is
; redundant.  No andq should appear before the first shrq.
define i64 @demand_through_xor_chain(i64 %x0) {
; CHECK-LABEL: demand_through_xor_chain:
; CHECK-NOT:   andq
; CHECK:       shrq $8
; CHECK:       andl $15
; CHECK-NEXT:  retq
  %a  = and i64 %x0, -256
  %s1 = lshr i64 %a, 8
  %x1 = xor i64 %s1, %x0
  %s2 = lshr i64 %x1, 4
  %x2 = xor i64 %s2, %x1
  %r  = and i64 %x2, 15
  ret i64 %r
}
