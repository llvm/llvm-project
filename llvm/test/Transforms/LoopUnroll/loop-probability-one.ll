; Check that a loop probability of one (indicating an always infinite loop) does
; not crash or otherwise break LoopUnroll behavior when it tries to compute new
; probabilities from it.
;
; That case indicates an always infinite loop.  A remainder loop cannot be
; calculated at run time when the original loop is infinite as infinity %
; UnrollCount is undefined, so consistent remainder loop probabilities are
; difficult or impossible to reason about.  The implementation chooses
; probabilities indicating that all remainder loop iterations will always
; execute.

; DEFINE: %{unroll} = opt < %s -unroll-count=3 -passes=loop-unroll -S
; DEFINE: %{rt} = %{unroll} -unroll-runtime

; RUN: %{unroll} | FileCheck %s -check-prefix UNROLL
; RUN: %{rt} -unroll-runtime-epilog=true | FileCheck %s -check-prefix EPILOG
; RUN: %{rt} -unroll-runtime-epilog=false | FileCheck %s -check-prefix PROLOG

define void @test(i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %inc = add i32 %i, 1
  %c = icmp slt i32 %inc, %n
  br i1 %c, label %loop, label %end, !prof !0

end:
  ret void
}


!0 = !{!"branch_weights", i32 1, i32 0}

; UNROLL: define void @test(i32 %n) {
; UNROLL: entry:
; UNROLL:   br label %loop
; UNROLL: loop:
; UNROLL:   br i1 %c, label %loop.1, label %end, !prof !0
; UNROLL: loop.1:
; UNROLL:   br i1 %c.1, label %loop.2, label %end, !prof !0
; UNROLL: loop.2:
; UNROLL:   br i1 %c.2, label %loop, label %end, !prof !0, !llvm.loop !1
; UNROLL-NOT: loop.3
; UNROLL: end:
; UNROLL:   ret void
; UNROLL: }
;
; Infinite unrolled loop.
; UNROLL: !0 = !{!"branch_weights", i32 1, i32 0}

; EPILOG: define void @test(i32 %n) {
; EPILOG: entry:
; EPILOG:   br i1 %{{.*}}, label %loop.epil.preheader, label %entry.new, !prof !0
; EPILOG: entry.new:
; EPILOG:   br label %loop
; EPILOG: loop:
; EPILOG:   br i1 %{{.*}}, label %loop, label %end.unr-lcssa, !prof !1
; EPILOG: end.unr-lcssa:
; EPILOG:   br i1 %{{.*}}, label %loop.epil.preheader, label %end, !prof !1
; EPILOG: loop.epil.preheader:
; EPILOG:   br label %loop.epil
; EPILOG: loop.epil:
; EPILOG:   br i1 %{{.*}}, label %loop.epil, label %end.epilog-lcssa, !prof !4
; EPILOG: end.epilog-lcssa:
; EPILOG:   br label %end
; EPILOG: end:
; EPILOG:   ret void
; EPILOG: }
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

; PROLOG: define void @test(i32 %n) {
; PROLOG: entry:
; PROLOG:   br i1 %{{.*}}, label %loop.prol.preheader, label %loop.prol.loopexit, !prof !0
; PROLOG: loop.prol.preheader:
; PROLOG:   br label %loop.prol
; PROLOG: loop.prol:
; PROLOG:   br i1 %{{.*}}, label %loop.prol, label %loop.prol.loopexit.unr-lcssa, !prof !1
; PROLOG: loop.prol.loopexit.unr-lcssa:
; PROLOG:   br label %loop.prol.loopexit
; PROLOG: loop.prol.loopexit:
; PROLOG:   br i1 %{{.*}}, label %end, label %entry.new, !prof !0
; PROLOG: entry.new:
; PROLOG:   br label %loop
; PROLOG: loop:
; PROLOG:   br i1 %{{.*}}, label %loop, label %end.unr-lcssa, !prof !4
; PROLOG: end.unr-lcssa:
; PROLOG:   br label %end
; PROLOG: end:
; PROLOG:   ret void
; PROLOG: }
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
