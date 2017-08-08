; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

; This code causes multiple endloop instructions to be generated for the
; same loop. The findLoopInstr would encounter for one endloop would encounter
; the other endloop, and return null in response. This resulted in a crash.
;
; Check that with the fix we are able to compile this code successfully.

target triple = "hexagon"

; Function Attrs: norecurse
define void @fred() local_unnamed_addr #0 align 2 {
b0:
  br label %b7

b1:                                               ; preds = %b9
  br i1 undef, label %b4, label %b2

b2:                                               ; preds = %b1
  %v3 = sub i32 undef, undef
  br label %b4

b4:                                               ; preds = %b2, %b1
  %v5 = phi i32 [ undef, %b1 ], [ %v3, %b2 ]
  br i1 undef, label %b14, label %b6

b6:                                               ; preds = %b4
  br label %b10

b7:                                               ; preds = %b0
  br i1 undef, label %b9, label %b8

b8:                                               ; preds = %b7
  unreachable

b9:                                               ; preds = %b7
  br label %b1

b10:                                              ; preds = %b21, %b6
  %v11 = phi i32 [ %v22, %b21 ], [ %v5, %b6 ]
  br i1 undef, label %b21, label %b12

b12:                                              ; preds = %b10
  br label %b15

b13:                                              ; preds = %b21
  br label %b14

b14:                                              ; preds = %b13, %b4
  ret void

b15:                                              ; preds = %b12
  br i1 undef, label %b16, label %b17

b16:                                              ; preds = %b15
  store i32 0, i32* undef, align 4
  br label %b21

b17:                                              ; preds = %b15
  br label %b18

b18:                                              ; preds = %b17
  br i1 undef, label %b19, label %b20

b19:                                              ; preds = %b18
  br label %b21

b20:                                              ; preds = %b18
  store i32 0, i32* undef, align 4
  br label %b21

b21:                                              ; preds = %b20, %b19, %b16, %b10
  %v22 = add i32 %v11, -8
  %v23 = icmp eq i32 %v22, 0
  br i1 %v23, label %b13, label %b10
}

attributes #0 = { norecurse "target-cpu"="hexagonv60" "target-features"="-hvx,-hvx-double,-long-calls" }
