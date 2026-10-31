; RUN: llc -mtriple=bpfel -mcpu=v4 < %s | FileCheck %s
;
; The verifier rejects a read of a register that was not written.
; An undefined value that ends up in a register is zero.
;
; A scalar poison is ISD::POISON. The value in @libcall is ISD::UNDEF.

declare void @use(i64, i64)
declare void @use32(i32)
declare i64 @get()

define void @arg(i64 %a) {
; CHECK-LABEL: arg:
; CHECK:       r2 = 0
; CHECK-NEXT:  call use
  call void @use(i64 %a, i64 poison)
  ret void
}

define void @arg32() {
; CHECK-LABEL: arg32:
; CHECK:       w1 = 0
; CHECK-NEXT:  call use32
  call void @use32(i32 poison)
  ret void
}

define i64 @ret() {
; CHECK-LABEL: ret:
; CHECK:       r0 = 0
; CHECK-NEXT:  exit
  ret i64 poison
}

define void @phi(i1 %c) {
; CHECK-LABEL: phi:
; CHECK:       r2 = 0
; CHECK:       call get
; CHECK:       r2 = r0
; CHECK:       call use
entry:
  br i1 %c, label %then, label %join
then:
  %x = call i64 @get()
  br label %join
join:
  %v = phi i64 [ poison, %entry ], [ %x, %then ]
  call void @use(i64 3, i64 %v)
  ret void
}

; Bits 0-63 of the product are used. They don't depend on the upper half
; of %b, which is passed in r4.
define i64 @libcall(i128 %a, i64 %b) {
; CHECK-LABEL: libcall:
; CHECK:       r4 = 0
; CHECK:       call __multi3
  %e = zext i64 %b to i128
  %m = mul i128 %a, %e
  %s = lshr i128 %m, 1
  %t = trunc i128 %s to i64
  %r = and i64 %t, 4611686018427387903
  ret i64 %r
}
