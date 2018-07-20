; RUN: llc -enable-aa-sched-mi -march=hexagon -mcpu=hexagonv5 -rdf-opt=0 \
; RUN:      < %s | FileCheck %s

; CHECK: {
; CHECK: = memd([[REG0:(r[0-9]+)]]++#8)
; CHECK-NOT: memw([[REG0]]+#0) =
; CHECK: }

define void @main() #0 {
cond.end.6:
  store i32 -1, i32* undef, align 8, !tbaa !0
  br label %polly.stmt.for.body.i

if.then:
  unreachable

if.end:
  ret void

polly.stmt.for.body.i24:
  %0 = extractelement <2 x i32> %add.ip_vec, i32 1
  br i1 undef, label %if.end, label %if.then

polly.stmt.for.body.i:
  %add.ip_vec30 = phi <2 x i32> [ %add.ip_vec, %polly.stmt.for.body.i ], [ zeroinitializer, %cond.end.6 ]
  %scevgep.phi = phi i32* [ %scevgep.inc, %polly.stmt.for.body.i ], [ undef, %cond.end.6 ]
  %polly.indvar = phi i32 [ %polly.indvar_next, %polly.stmt.for.body.i ], [ 0, %cond.end.6 ]
  %vector_ptr = bitcast i32* %scevgep.phi to <2 x i32>*
  %_p_vec_full = load <2 x i32>, <2 x i32>* %vector_ptr, align 8
  %add.ip_vec = add <2 x i32> %_p_vec_full, %add.ip_vec30
  %polly.indvar_next = add nsw i32 %polly.indvar, 2
  %polly.loop_cond = icmp slt i32 %polly.indvar, 4
  %scevgep.inc = getelementptr i32, i32* %scevgep.phi, i32 2
  br i1 %polly.loop_cond, label %polly.stmt.for.body.i, label %polly.stmt.for.body.i24
}

attributes #0 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

!0 = !{!"int", !1}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
