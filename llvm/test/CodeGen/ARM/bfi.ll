; RUN: llc -mtriple=arm -mattr=+v6t2 %s -o - | FileCheck %s

%struct.F = type { [3 x i8], i8 }

@X = common global %struct.F zeroinitializer, align 4 ; <%struct.F*> [#uses=1]

define void @f1([1 x i32] %f.coerce0) nounwind {
entry:
; CHECK: f1
; CHECK: mov r2, #10
; CHECK: bfi r1, r2, #22, #4
  %0 = load i32, i32* bitcast (%struct.F* @X to i32*), align 4 ; <i32> [#uses=1]
  %1 = and i32 %0, -62914561                      ; <i32> [#uses=1]
  %2 = or i32 %1, 41943040                        ; <i32> [#uses=1]
  store i32 %2, i32* bitcast (%struct.F* @X to i32*), align 4
  ret void
}

define i32 @f2(i32 %A, i32 %B) nounwind {
entry:
; CHECK: f2
; CHECK: lsr{{.*}}#7
; CHECK: bfi r0, r1, #7, #16
  %and = and i32 %A, -8388481                     ; <i32> [#uses=1]
  %and2 = and i32 %B, 8388480                     ; <i32> [#uses=1]
  %or = or i32 %and2, %and                        ; <i32> [#uses=1]
  ret i32 %or
}

define i32 @f3(i32 %A, i32 %B) nounwind {
entry:
; CHECK: f3
; CHECK: lsr{{.*}} #7
; CHECK: bfi {{.*}}, #7, #16
  %and = and i32 %A, 8388480                      ; <i32> [#uses=1]
  %and2 = and i32 %B, -8388481                    ; <i32> [#uses=1]
  %or = or i32 %and2, %and                        ; <i32> [#uses=1]
  ret i32 %or
}

; rdar://8752056
define i32 @f4(i32 %a) nounwind {
; CHECK: f4
; CHECK: movw [[R1:r[0-9]+]], #3137
; CHECK: bfi [[R1]], {{r[0-9]+}}, #15, #5
  %1 = shl i32 %a, 15
  %ins7 = and i32 %1, 1015808
  %ins12 = or i32 %ins7, 3137
  ret i32 %ins12
}

; rdar://8458663
define i32 @f5(i32 %a, i32 %b) nounwind {
entry:
; CHECK-LABEL: f5:
; CHECK-NOT: bfc
; CHECK: bfi r0, r1, #20, #4
  %0 = and i32 %a, -15728641
  %1 = shl i32 %b, 20
  %2 = and i32 %1, 15728640
  %3 = or i32 %2, %0
  ret i32 %3
}

; rdar://9609030
define i32 @f6(i32 %a, i32 %b) nounwind readnone {
entry:
; CHECK-LABEL: f6:
; CHECK-NOT: bic
; CHECK: bfi r0, r1, #8, #9
  %and = and i32 %a, -130817
  %and2 = shl i32 %b, 8
  %shl = and i32 %and2, 130816
  %or = or i32 %shl, %and
  ret i32 %or
}

define i32 @f7(i32 %x, i32 %y) {
; CHECK-LABEL: f7:
; CHECK: bfi r1, r0, #4, #1
  %y2 = and i32 %y, 4294967040 ; 0xFFFFFF00
  %and = and i32 %x, 4
  %or = or i32 %y2, 16
  %cmp = icmp ne i32 %and, 0
  %sel = select i1 %cmp, i32 %or, i32 %y2
  ret i32 %sel
}

define i32 @f8(i32 %x, i32 %y) {
; CHECK-LABEL: f8:
; CHECK: bfi r1, r0, #4, #1
; CHECK: bfi r1, r0, #5, #1
  %y2 = and i32 %y, 4294967040 ; 0xFFFFFF00
  %and = and i32 %x, 4
  %or = or i32 %y2, 48
  %cmp = icmp ne i32 %and, 0
  %sel = select i1 %cmp, i32 %or, i32 %y2
  ret i32 %sel
}

define i32 @f9(i32 %x, i32 %y) {
; CHECK-LABEL: f9:
; CHECK-NOT: bfi
  %y2 = and i32 %y, 4294967040 ; 0xFFFFFF00
  %and = and i32 %x, 4
  %or = or i32 %y2, 48
  %cmp = icmp ne i32 %and, 0
  %sel = select i1 %cmp, i32 %y2, i32 %or
  ret i32 %sel
}

define i32 @f10(i32 %x, i32 %y) {
; CHECK-LABEL: f10:
; CHECK: bfi r1, r0, #4, #2
  %y2 = and i32 %y, 4294967040 ; 0xFFFFFF00
  %and = and i32 %x, 4
  %or = or i32 %y2, 32
  %cmp = icmp ne i32 %and, 0
  %sel = select i1 %cmp, i32 %or, i32 %y2

  %aand = and i32 %x, 2
  %aor = or i32 %sel, 16
  %acmp = icmp ne i32 %aand, 0
  %asel = select i1 %acmp, i32 %aor, i32 %sel

  ret i32 %asel
}

define i32 @f11(i32 %x, i32 %y) {
; CHECK-LABEL: f11:
; CHECK: bfi r1, r0, #4, #3
  %y2 = and i32 %y, 4294967040 ; 0xFFFFFF00
  %and = and i32 %x, 4
  %or = or i32 %y2, 32
  %cmp = icmp ne i32 %and, 0
  %sel = select i1 %cmp, i32 %or, i32 %y2

  %aand = and i32 %x, 2
  %aor = or i32 %sel, 16
  %acmp = icmp ne i32 %aand, 0
  %asel = select i1 %acmp, i32 %aor, i32 %sel

  %band = and i32 %x, 8
  %bor = or i32 %asel, 64
  %bcmp = icmp ne i32 %band, 0
  %bsel = select i1 %bcmp, i32 %bor, i32 %asel

  ret i32 %bsel
}

define i32 @f12(i32 %x, i32 %y) {
; CHECK-LABEL: f12:
; CHECK: bfi r1, r0, #4, #1
  %y2 = and i32 %y, 4294967040 ; 0xFFFFFF00
  %and = and i32 %x, 4
  %or = or i32 %y2, 16
  %cmp = icmp eq i32 %and, 0
  %sel = select i1 %cmp, i32 %y2, i32 %or
  ret i32 %sel
}

define i32 @f13(i32 %x, i32 %y) {
; CHECK-LABEL: f13:
; CHECK-NOT: bfi
  %y2 = and i32 %y, 4294967040 ; 0xFFFFFF00
  %and = and i32 %x, 4
  %or = or i32 %y2, 16
  %cmp = icmp eq i32 %and, 42 ; Not comparing against zero!
  %sel = select i1 %cmp, i32 %y2, i32 %or
  ret i32 %sel
}
