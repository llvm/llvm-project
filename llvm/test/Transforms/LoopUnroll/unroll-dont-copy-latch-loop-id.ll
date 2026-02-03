; Check that !llvm.loop is not copied from the original loop's latch to the
; corresponding non-latch branch in any unrolled iteration.

; DEFINE: %{unroll} = opt < %s -passes=loop-unroll -S
; DEFINE: %{fc} = FileCheck %s -implicit-check-not='!llvm.loop' -check-prefixes

; Check partial unroll: only the unrolled loop's latch has !llvm.loop.
; RUN: %{unroll} -unroll-count=3 | %{fc} UR3
; UR3: br i1 %c.2, label %body, label %exit, !llvm.loop !0
; UR3: !0 = distinct !{!0, !1, !2}
; UR3: !1 = !{!"copied"}
; UR3: !2 = !{!"llvm.loop.unroll.disable"}

; Check complete unroll: no !llvm.loop remains because no loop remains.
; RUN: %{unroll} -unroll-count=4 | %{fc} UR4
; UR4: br label %exit

; Check runtime cases: unrolled and remainder loops' latches have !llvm.loop.
; DEFINE: %{rt} = %{unroll} -unroll-count=3 -unroll-runtime
; RUN: %{rt} -unroll-runtime-epilog=true | %{fc} EPILOG
; RUN: %{rt} -unroll-runtime-epilog=false | %{fc} PROLOG
;
; EPILOG: br i1 %{{.*}}, label %body, label %exit.unr-lcssa, !llvm.loop !0
; EPILOG: br i1 {{.*}}, label %body.epil, label %exit.epilog-lcssa, !llvm.loop !3
; EPILOG: !0 = distinct !{!0, !1, !2}
; EPILOG: !1 = !{!"copied"}
; EPILOG: !2 = !{!"llvm.loop.unroll.disable"}
; EPILOG: !3 = distinct !{!3, !2}
;
; PROLOG: br i1 %{{.*}}, label %body.prol, label %body.prol.loopexit.unr-lcssa, !llvm.loop !0
; PROLOG: br i1 %{{.*}}, label %body, label %exit.unr-lcssa, !llvm.loop !2
; PROLOG: !0 = distinct !{!0, !1}
; PROLOG: !1 = !{!"llvm.loop.unroll.disable"}
; PROLOG: !2 = distinct !{!2, !3, !1}
; PROLOG: !3 = !{!"copied"}

define void @test(i32 %n) {
entry:
  %max = call i32 @llvm.umin.i32(i32 %n, i32 3)
  br label %body

body:
  %i = phi i32 [ 0, %entry ], [ %inc, %body ]
  %inc = add i32 %i, 1
  %c = icmp ult i32 %i, %max
  br i1 %c, label %body, label %exit, !llvm.loop !0

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"copied"}
