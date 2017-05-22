; RUN: opt -early-cse -S %s | FileCheck %s

%mystruct = type { i32 }

; @var is global so that *every* GEP argument is Constant.
@var = external global %mystruct

; Control flow is to make the dominance tree consider the final icmp before it
; gets to simplify the purely constant one (%tst). Since that icmp uses the
; select that gets considered next. Finally the select simplification looks at
; the %tst icmp and we don't want it to speculate about what happens if "i32 0"
; is actually "i32 1", broken universes are automatic UB.
;
; In this case doing the speculation would create an invalid GEP(@var, 0, 1) and
; crash.

define i1 @test_constant_speculation() {
; CHECK-LABEL: define i1 @test_constant_speculation
entry:
  br i1 undef, label %end, label %select

select:
; CHECK: select:
; CHECK-NOT: icmp
; CHECK-NOT: getelementptr
; CHECK-NOT: select

  %tst = icmp eq i32 1, 0
  %elt = getelementptr %mystruct, %mystruct* @var, i64 0, i32 0
  %sel = select i1 %tst, i32* null, i32* %elt
  br label %end

end:
; CHECK: end:
; CHECK: %tmp = phi i32* [ null, %entry ], [ getelementptr inbounds (%mystruct, %mystruct* @var, i64 0, i32 0), %select ]
  %tmp = phi i32* [null, %entry], [%sel, %select]
  %res = icmp eq i32* %tmp, null
  ret i1 %res
}
