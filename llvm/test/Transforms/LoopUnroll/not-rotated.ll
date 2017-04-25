; PR28103
; Bail out if the two successors are not the header
; and another bb outside of the loop. This case is not
; properly handled by LoopUnroll, currently.

; RUN: opt -loop-unroll -verify-dom-info %s
; REQUIRE: asserts

define void @tinkywinky(i1 %patatino) {
entry:
  br label %header1
header1:
  %indvars.iv = phi i64 [ 1, %body2 ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 1
  br i1 %exitcond, label %body1, label %exit
body1:
  br i1 %patatino, label %body2, label %sink
body2:
  br i1 %patatino, label %header1, label %body3
body3:
  br label %sink
sink:
  br label %body2
exit:
  ret void
}
