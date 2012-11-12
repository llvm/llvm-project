; RUN: llc -march=thumb -mattr=+v6t2 < %s | FileCheck %s

%struct.F = type { [3 x i8], i8 }

@X = common global %struct.F zeroinitializer, align 4 ; <%struct.F*> [#uses=1]

define void @f1([1 x i32] %f.coerce0) nounwind {
entry:
; CHECK: f1
; CHECK: movs r2, #10
; CHECK: bfi r1, r2, #22, #4
  %0 = load i32* bitcast (%struct.F* @X to i32*), align 4 ; <i32> [#uses=1]
  %1 = and i32 %0, -62914561                      ; <i32> [#uses=1]
  %2 = or i32 %1, 41943040                        ; <i32> [#uses=1]
  store i32 %2, i32* bitcast (%struct.F* @X to i32*), align 4
  ret void
}

define i32 @f2(i32 %A, i32 %B) nounwind readnone optsize {
entry:
; CHECK: f2
; CHECK: lsrs  r1, r1, #7
; CHECK: bfi r0, r1, #7, #16
  %and = and i32 %A, -8388481                     ; <i32> [#uses=1]
  %and2 = and i32 %B, 8388480                     ; <i32> [#uses=1]
  %or = or i32 %and2, %and                        ; <i32> [#uses=1]
  ret i32 %or
}

define i32 @f3(i32 %A, i32 %B) nounwind readnone optsize {
entry:
; CHECK: f3
; CHECK: lsrs {{.*}}, #7
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
; CHECK: bfi [[R1]], {{.*}}, #15, #5
  %1 = shl i32 %a, 15
  %ins7 = and i32 %1, 1015808
  %ins12 = or i32 %ins7, 3137
  ret i32 %ins12
}

; rdar://9177502
define i32 @f5(i32 %a, i32 %b) nounwind readnone {
entry:
; CHECK: f5
; CHECK-NOT: bfi r0, r2, #0, #1
%and = and i32 %a, 2
%b.masked = and i32 %b, -2
%and3 = or i32 %b.masked, %and
ret i32 %and3
}
