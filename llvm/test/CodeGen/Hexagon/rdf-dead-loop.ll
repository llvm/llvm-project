; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK-NOT: ={{.*}}add
; CHECK-NOT: mem{{[bdhwu]}}

define void @main() #0 {
entry:
  br label %body

body:
  %ip_vec30 = phi <2 x i32> [ %ip_vec, %body ], [ zeroinitializer, %entry ]
  %scevgep.phi = phi i32* [ %scevgep.inc, %body ], [ undef, %entry ]
  %polly.indvar = phi i32 [ %polly.indvar_next, %body ], [ 0, %entry ]
  %vector_ptr = bitcast i32* %scevgep.phi to <2 x i32>*
  %_p_vec_full = load <2 x i32>, <2 x i32>* %vector_ptr, align 8
  %ip_vec = add <2 x i32> %_p_vec_full, %ip_vec30
  %polly.indvar_next = add nsw i32 %polly.indvar, 2
  %polly.loop_cond = icmp slt i32 %polly.indvar, 4
  %scevgep.inc = getelementptr i32, i32* %scevgep.phi, i32 2
  br i1 %polly.loop_cond, label %body, label %exit

exit:
  %0 = extractelement <2 x i32> %ip_vec, i32 1
  ret void

}

attributes #0 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

!0 = !{!"int", !1}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
