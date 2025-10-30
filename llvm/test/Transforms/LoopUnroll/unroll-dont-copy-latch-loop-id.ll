; Check that !llvm.loop is not copied from the original loop's latch to the
; corresponding non-latch branch in any unrolled iteration.

; The -implicit-check-not options make sure that no additional label or
; !llvm.loop shows up.
; DEFINE: %{unroll} = opt < %s -passes=loop-unroll -S
; DEFINE: %{fc} = FileCheck %s \
; DEFINE:     -implicit-check-not='{{^[^ ;]*:}}' \
; DEFINE:     -implicit-check-not='!llvm.loop' \
; DEFINE:     -check-prefixes

; Check partial unroll: only the unrolled loop's latch has !llvm.loop.
; RUN: %{unroll} -unroll-count=3 | %{fc} ALL,UR,UR3

; Check complete unroll: no !llvm.loop remains because no loop remains.
; RUN: %{unroll} -unroll-count=4 | %{fc} ALL,UR,UR4

; Check remainder: both loops have a !llvm.loop.
; DEFINE: %{rt} = %{unroll} -unroll-count=3 -unroll-runtime
; RUN: %{rt} -unroll-runtime-epilog=true | %{fc} ALL,RT,EPILOG
; RUN: %{rt} -unroll-runtime-epilog=false | %{fc} ALL,RT,PROLOG

;    ALL: define void @test(i32 %n) {
;    ALL: entry:
;     UR:   br label %body
; EPILOG:   br i1 %{{.*}}, label %body.epil.preheader, label %entry.new
; PROLOG:   br i1 %{{.*}}, label %body.prol.preheader, label %body.prol.loopexit
; PROLOG: body.prol.preheader:
; PROLOG:   br label %body.prol
; PROLOG: body.prol:
; PROLOG:   br i1 %{{.*}}, label %body.prol, label %body.prol.loopexit.unr-lcssa, !llvm.loop !0
; PROLOG: body.prol.loopexit.unr-lcssa:
; PROLOG:   br label %body.prol.loopexit
; PROLOG: body.prol.loopexit:
; PROLOG:   br i1 %{{.*}}, label %exit, label %entry.new
;     RT: entry.new:
;     RT:   br label %body
;    ALL: body:
;     UR:   br i1 %c, label %body.1, label %exit
; EPILOG:   br i1 %{{.*}}, label %body, label %exit.unr-lcssa, !llvm.loop !0
; PROLOG:   br i1 %{{.*}}, label %body, label %exit.unr-lcssa, !llvm.loop !2
;     UR: body.1:
;     UR:   br i1 %c.1, label %body.2, label %exit
;     UR: body.2:
;    UR3:   br i1 %c.2, label %body, label %exit, !llvm.loop !0
;    UR4:   br i1 %c.2, label %body.3, label %exit
;    UR4: body.3:
;    UR4:   br label %exit
;     RT: exit.unr-lcssa:
; EPILOG:   br i1 {{.*}}, label %body.epil.preheader, label %exit
; PROLOG:   br label %exit
; EPILOG: body.epil.preheader:
; EPILOG:   br label %body.epil
; EPILOG: body.epil:
; EPILOG:   br i1 {{.*}}, label %body.epil, label %exit.epilog-lcssa, !llvm.loop !3
; EPILOG: exit.epilog-lcssa:
; EPILOG:   br label %exit
;    ALL: exit:
;    ALL:   ret void
;    ALL: }
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

; UR3: !0 = distinct !{!0, !1, !2}
; UR3: !1 = !{!"copied"}
; UR3: !2 = !{!"llvm.loop.unroll.disable"}
;
; EPILOG: !0 = distinct !{!0, !1, !2}
; EPILOG: !1 = !{!"copied"}
; EPILOG: !2 = !{!"llvm.loop.unroll.disable"}
; EPILOG: !3 = distinct !{!3, !2}
;
; PROLOG: !0 = distinct !{!0, !1}
; PROLOG: !1 = !{!"llvm.loop.unroll.disable"}
; PROLOG: !2 = distinct !{!2, !3, !1}
; PROLOG: !3 = !{!"copied"}
!0 = distinct !{!0, !1}
!1 = !{!"copied"}
