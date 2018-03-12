; RUN: llc -O2 -march=hexagon -hexagon-expand-condsets=0 < %s
; REQUIRES: asserts
; Disable expand-condsets because it will assert on undefined registers.

target triple = "hexagon-unknown--elf"

%s.0 = type { %s.0*, %s.0* }
%s.1 = type { %s.1*, %s.1** }

@g0 = external global %s.0, align 4

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  br i1 undef, label %b2, label %b1

b1:                                               ; preds = %b0
  unreachable

b2:                                               ; preds = %b0
  br i1 undef, label %b26, label %b3

b3:                                               ; preds = %b2
  br i1 undef, label %b6, label %b4

b4:                                               ; preds = %b3
  br i1 undef, label %b5, label %b26

b5:                                               ; preds = %b4
  br i1 undef, label %b7, label %b26

b6:                                               ; preds = %b3
  br label %b7

b7:                                               ; preds = %b6, %b5
  br i1 undef, label %b11, label %b8

b8:                                               ; preds = %b7
  br i1 undef, label %b10, label %b9

b9:                                               ; preds = %b8
  unreachable

b10:                                              ; preds = %b8
  unreachable

b11:                                              ; preds = %b7
  br i1 undef, label %b25, label %b12

b12:                                              ; preds = %b11
  br i1 undef, label %b14, label %b13

b13:                                              ; preds = %b12
  br label %b14

b14:                                              ; preds = %b13, %b12
  br i1 undef, label %b15, label %b16

b15:                                              ; preds = %b14
  br label %b16

b16:                                              ; preds = %b15, %b14
  br i1 undef, label %b18, label %b17

b17:                                              ; preds = %b16
  unreachable

b18:                                              ; preds = %b16
  %v0 = load %s.0*, %s.0** getelementptr inbounds (%s.0, %s.0* @g0, i32 0, i32 1), align 4
  %v1 = load %s.0*, %s.0** getelementptr inbounds (%s.0, %s.0* @g0, i32 0, i32 0), align 4
  %v2 = select i1 undef, %s.0* %v0, %s.0* %v1
  br i1 undef, label %b22, label %b19

b19:                                              ; preds = %b18
  %v3 = load %s.1*, %s.1** undef, align 4
  %v4 = icmp eq %s.1* %v3, null
  br i1 %v4, label %b21, label %b20

b20:                                              ; preds = %b19
  store %s.1** undef, %s.1*** undef, align 4
  br label %b21

b21:                                              ; preds = %b20, %b19
  br label %b22

b22:                                              ; preds = %b21, %b18
  br i1 undef, label %b24, label %b23

b23:                                              ; preds = %b22
  store %s.0* %v2, %s.0** undef, align 4
  br label %b24

b24:                                              ; preds = %b23, %b22
  unreachable

b25:                                              ; preds = %b11
  unreachable

b26:                                              ; preds = %b5, %b4, %b2
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
