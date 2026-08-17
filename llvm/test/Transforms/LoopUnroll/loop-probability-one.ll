; Check that a loop probability of one (indicating an always infinite loop) does
; not crash or otherwise break LoopUnroll behavior when it tries to compute new
; probabilities from it.

; DEFINE: %{unroll} = opt < %t.ll -unroll-count=3 -passes=loop-unroll -S
; DEFINE: %{fc} = FileCheck %s \
; DEFINE:     -implicit-check-not='llvm.loop.estimated_trip_count' \
; DEFINE:     -implicit-check-not='!prof' \
; DEFINE:     -implicit-check-not='branch_weights' \
; DEFINE:     -implicit-check-not='call void @f' -check-prefixes

; ------------------------------------------------------------------------------
; A partially unrolled loop remains infinite.
;
; RUN: sed -e s/@N@/%n/ %s > %t.ll
; RUN: %{unroll} | %{fc} PART-ALL-COND
;
; PART-ALL-COND: call void @f
; PART-ALL-COND: br i1 %{{.*}}, label %loop.1, label %end, !prof !0
; PART-ALL-COND: call void @f
; PART-ALL-COND: br i1 %{{.*}}, label %loop.2, label %end, !prof !0
; PART-ALL-COND: call void @f
; PART-ALL-COND: br i1 %{{.*}}, label %loop, label %end, !prof !0, !llvm.loop !1
; PART-ALL-COND: !0 = !{!"branch_weights", i32 1, i32 0}

; ------------------------------------------------------------------------------
; A partially unrolled loop remains infinite even if some iterations' latches
; become unconditional.
;
; RUN: sed -e s/@N@/5/ %s > %t.ll
; RUN: %{unroll} | %{fc} PART-SOME-COND
;
; PART-SOME-COND:     call void @f
; PART-SOME-COND-NOT: br
; PART-SOME-COND:     call void @f
; PART-SOME-COND:     br i1 %{{.*}}, label %loop.2, label %end, !prof !0
; PART-SOME-COND:     call void @f
; PART-SOME-COND:     br label %loop, !llvm.loop !1
; PART-SOME-COND:     !0 = !{!"branch_weights", i32 1, i32 0}

; ------------------------------------------------------------------------------
; A completely unrolled loop cannot be infinite, so consistent unrolled loop
; probabilities are impossible.  The implementation chooses probabilities
; indicating that all unrolled loop iterations will always execute.
;
; RUN: sed -e s/@N@/%max3/ %s > %t.ll
; RUN: %{unroll} | %{fc} COMPLETE-SOME-COND
;
; COMPLETE-SOME-COND: call void @f
; COMPLETE-SOME-COND: br i1 %{{.*}}, label %loop.1, label %end, !prof !0
; COMPLETE-SOME-COND: call void @f
; COMPLETE-SOME-COND: br i1 %{{.*}}, label %loop.2, label %end, !prof !0
; COMPLETE-SOME-COND: call void @f
; COMPLETE-SOME-COND: br label %end
; COMPLETE-SOME-COND: !0 = !{!"branch_weights", i32 1, i32 0}

; ------------------------------------------------------------------------------
; A completely unrolled loop with no remaining conditional latches gives the
; implementation no probabilities to set.  Check that it still behaves.
;
; RUN: sed -e s/@N@/3/ %s > %t.ll
; RUN: %{unroll} | %{fc} COMPLETE-NO-COND
;
; COMPLETE-NO-COND:     call void @f
; COMPLETE-NO-COND-NOT: br
; COMPLETE-NO-COND:     call void @f
; COMPLETE-NO-COND-NOT: br
; COMPLETE-NO-COND:     call void @f

; ------------------------------------------------------------------------------
; A remainder loop cannot be calculated at run time when the original loop is
; infinite as infinity % UnrollCount is undefined, so consistent remainder loop
; probabilities are difficult or impossible to reason about.  The implementation
; chooses probabilities indicating that all remainder loop iterations will
; always execute.
;
; RUN: sed -e s/@N@/%n/ %s > %t.ll
; DEFINE: %{rt} = %{unroll} -unroll-runtime
; RUN: %{rt} -unroll-runtime-epilog=true | %{fc} EPILOG
; RUN: %{rt} -unroll-runtime-epilog=false | %{fc} PROLOG
;
; Unrolled loop guard, body, and latch.
; EPILOG:     br i1 %{{.*}}, label %loop.epil.preheader, label %entry.new, !prof !0
; EPILOG:     call void @f
; EPILOG-NOT: br
; EPILOG:     call void @f
; EPILOG-NOT: br
; EPILOG:     call void @f
; EPILOG:     br i1 %{{.*}}, label %loop, label %end.unr-lcssa, !prof !1
;
; Epilogue guard, body, and latch.
; EPILOG: br i1 %{{.*}}, label %loop.epil.preheader, label %end, !prof !1
; EPILOG: call void @f
; EPILOG: br i1 %{{.*}}, label %loop.epil, label %end.epilog-lcssa, !prof !4
;
; Unrolled loop guard: Unrolled loop is always entered.
; EPILOG: !0 = !{!"branch_weights", i32 0, i32 -2147483648}
;
; Unrolled loop latch: Unrolled loop is infinite.
; Epilogue loop guard: Epilogue loop is always entered if unrolled loop exits.
; EPILOG: !1 = !{!"branch_weights", i32 -2147483648, i32 0}
;
; Epilogue loop latch: Epilogue loop executes both of its 2 iterations.
; EPILOG: !4 = !{!"branch_weights", i32 1073741824, i32 1073741824}
;
; Prologue guard, body, and latch.
; PROLOG: br i1 %{{.*}}, label %loop.prol.preheader, label %loop.prol.loopexit, !prof !0
; PROLOG: call void @f
; PROLOG: br i1 %{{.*}}, label %loop.prol, label %loop.prol.loopexit.unr-lcssa, !prof !1
;
; Unrolled loop guard, body, and latch.
; PROLOG:     br i1 %{{.*}}, label %end, label %entry.new, !prof !0
; PROLOG:     call void @f
; PROLOG-NOT: br
; PROLOG:     call void @f
; PROLOG-NOT: br
; PROLOG:     call void @f
; PROLOG:     br i1 %{{.*}}, label %loop, label %end.unr-lcssa, !prof !4
;
; FIXME: Branch weights still need to be fixed in the case of prologues (issue
; #135812), so !0 and !1 do not yet match their comments below.  When we do
; fix it, this test will hopefully catch any bug like issue #165998, which
; impacted the case of epilogues.
;
; Prologue loop guard: Prologue loop is always entered.
; Unrolled loop guard: Unrolled loop is always entered.
; PROLOG: !0 = !{!"branch_weights", i32 1, i32 127}
;
; Prologue loop latch: Prologue loop executes both of its 2 iterations.
; PROLOG: !1 = !{!"branch_weights", i32 0, i32 1}
;
; Unrolled loop latch: Unrolled loop is infinite.
; PROLOG: !4 = !{!"branch_weights", i32 1, i32 0}

declare void @f(i32)

define void @test(i32 %n) {
entry:
  %max3 = call i32 @llvm.umin.i32(i32 %n, i32 3)
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  call void @f(i32 %i)
  %inc = add i32 %i, 1
  %c = icmp slt i32 %inc, @N@
  br i1 %c, label %loop, label %end, !prof !0

end:
  ret void
}

!0 = !{!"branch_weights", i32 1, i32 0}
