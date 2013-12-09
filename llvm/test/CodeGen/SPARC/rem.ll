; RUN: llc < %s -march=sparcv9 | FileCheck %s

; CHECK-LABEL: test1:
; CHECK:        sdivx %o0, %o1, %o2
; CHECK-NEXT:   mulx %o2, %o1, %o1
; CHECK-NEXT:   jmp %o7+8
; CHECK-NEXT:   sub %o0, %o1, %o0

define i64 @test1(i64 %X, i64 %Y) {
        %tmp1 = srem i64 %X, %Y
        ret i64 %tmp1
}

; CHECK-LABEL: test2:
; CHECK:        udivx %o0, %o1, %o2
; CHECK-NEXT:   mulx %o2, %o1, %o1
; CHECK-NEXT:   jmp %o7+8
; CHECK-NEXT:   sub %o0, %o1, %o0

define i64 @test2(i64 %X, i64 %Y) {
        %tmp1 = urem i64 %X, %Y
        ret i64 %tmp1
}

; PR18150
; CHECK-LABEL: test3
; CHECK:       sethi 2545, [[R0:%[gilo][0-7]]]
; CHECK:       or    [[R0]], 379, [[R1:%[gilo][0-7]]]
; CHECK:       mulx  %o0, [[R1]], [[R2:%[gilo][0-7]]]
; CHECK:       udivx [[R2]], 1021, [[R3:%[gilo][0-7]]]
; CHECK:       mulx  [[R3]], 1021, [[R4:%[gilo][0-7]]]
; CHECK:       sub   [[R2]], [[R4]], %o0

define i64 @test3(i64 %b) {
entry:
  %mul = mul i64 %b, 2606459
  %rem = urem i64 %mul, 1021
  ret i64 %rem
}
