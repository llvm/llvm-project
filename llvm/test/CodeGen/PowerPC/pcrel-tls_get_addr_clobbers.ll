; RUN: llc -verify-machineinstrs -mtriple="powerpc64le-unknown-linux-gnu" \
; RUN:  -ppc-asm-full-reg-names -mcpu=pwr10 -relocation-model=pic < %s | FileCheck %s

%0 = type { ptr, ptr }
@x = external dso_local thread_local unnamed_addr global ptr, align 8
define void @test(ptr %arg) {
; CHECK-LABEL: test:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    mflr r0
; CHECK:         std r30, -16(r1)
; CHECK-NEXT:    std r0, 16(r1)
; CHECK-NEXT:    stdu r1, -48(r1)
; CHECK-NEXT:    mr r30, r3
; CHECK-NEXT:    paddi r3, 0, x@got@tlsld@pcrel, 1
; CHECK-NEXT:    bl __tls_get_addr@notoc(x@tlsld)
; CHECK-NEXT:    paddi r3, r3, x@DTPREL, 0
; CHECK-NEXT:    std r30, 0(r3)
; CHECK-NEXT:    addi r1, r1, 48
; CHECK-NEXT:    ld r0, 16(r1)
; CHECK-NEXT:    ld r30, -16(r1)
; CHECK-NEXT:    mtlr r0
entry:
  store ptr %arg, ptr @x, align 8
  ret void
}
