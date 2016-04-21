; This file verifies the behavior of the OptBisect class, which is used to
; diagnose optimization related failures.  The tests check various
; invocations that result in different sets of optimization passes that
; are run in different ways.
;
; Because the exact set of optimizations that will be run is expected to
; change over time, the checks for disabling passes are written in a
; conservative way that avoids assumptions about which specific passes
; will be disabled.

; RUN: opt -disable-output -disable-verify \
; RUN:     -passes=inferattrs -opt-bisect-limit=-1 %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-MODULE-PASS
; CHECK-MODULE-PASS: BISECT: running pass (1) InferFunctionAttrsPass on module

; RUN: opt -disable-output -disable-verify \
; RUN:     -passes=inferattrs -opt-bisect-limit=0 %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-LIMIT-MODULE-PASS
; CHECK-LIMIT-MODULE-PASS: BISECT: NOT running pass (1) InferFunctionAttrsPass on module

; RUN: opt -disable-output -disable-verify \
; RUN:     -passes=early-cse -opt-bisect-limit=-1 %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-FUNCTION-PASS
; CHECK-FUNCTION-PASS: BISECT: running pass (1) EarlyCSEPass on function (f1)
; CHECK-FUNCTION-PASS: BISECT: running pass (2) EarlyCSEPass on function (f2)
; CHECK-FUNCTION-PASS: BISECT: running pass (3) EarlyCSEPass on function (f3)

; RUN: opt -disable-output -disable-verify \
; RUN:     -passes=early-cse -opt-bisect-limit=2 %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-LIMIT-FUNCTION-PASS
; CHECK-LIMIT-FUNCTION-PASS: BISECT: running pass (1) EarlyCSEPass on function (f1)
; CHECK-LIMIT-FUNCTION-PASS: BISECT: running pass (2) EarlyCSEPass on function (f2)
; CHECK-LIMIT-FUNCTION-PASS: BISECT: NOT running pass (3) EarlyCSEPass on function (f3)

; RUN: opt -disable-output -disable-verify \
; RUN:     -passes=function-attrs -opt-bisect-limit=-1 %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-CGSCC-PASS
; CHECK-CGSCC-PASS: BISECT: running pass (1) PostOrderFunctionAttrsPass on SCC (f2)
; CHECK-CGSCC-PASS: BISECT: running pass (2) PostOrderFunctionAttrsPass on SCC (f3)
; CHECK-CGSCC-PASS: BISECT: running pass (3) PostOrderFunctionAttrsPass on SCC (f1)

; RUN: opt -disable-output -disable-verify \
; RUN:     -passes=function-attrs -opt-bisect-limit=2 %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-LIMIT-CGSCC-PASS
; CHECK-LIMIT-CGSCC-PASS: BISECT: running pass (1) PostOrderFunctionAttrsPass on SCC (f2)
; CHECK-LIMIT-CGSCC-PASS: BISECT: running pass (2) PostOrderFunctionAttrsPass on SCC (f3)
; CHECK-LIMIT-CGSCC-PASS: BISECT: NOT running pass (3) PostOrderFunctionAttrsPass on SCC (f1)

; RUN: opt -disable-output -disable-verify -opt-bisect-limit=-1 \
; RUN:     -passes='inferattrs,cgscc(function-attrs,function(early-cse))' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-MULTI-PASS
; CHECK-MULTI-PASS: BISECT: running pass (1) InferFunctionAttrsPass on module
; CHECK-MULTI-PASS: BISECT: running pass (2) PostOrderFunctionAttrsPass on SCC (f2)
; CHECK-MULTI-PASS: BISECT: running pass (3) EarlyCSEPass on function (f2)
; CHECK-MULTI-PASS: BISECT: running pass (4) PostOrderFunctionAttrsPass on SCC (f3)
; CHECK-MULTI-PASS: BISECT: running pass (5) EarlyCSEPass on function (f3)
; CHECK-MULTI-PASS: BISECT: running pass (6) PostOrderFunctionAttrsPass on SCC (f1)
; CHECK-MULTI-PASS: BISECT: running pass (7) EarlyCSEPass on function (f1)

; RUN: opt -disable-output -disable-verify -opt-bisect-limit=5 \
; RUN:     -passes='inferattrs,cgscc(function-attrs,function(early-cse))' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-LIMIT-MULTI-PASS
; CHECK-LIMIT-MULTI-PASS: BISECT: running pass (1) InferFunctionAttrsPass on module
; CHECK-LIMIT-MULTI-PASS: BISECT: running pass (2) PostOrderFunctionAttrsPass on SCC (f2)
; CHECK-LIMIT-MULTI-PASS: BISECT: running pass (3) EarlyCSEPass on function (f2)
; CHECK-LIMIT-MULTI-PASS: BISECT: running pass (4) PostOrderFunctionAttrsPass on SCC (f3)
; CHECK-LIMIT-MULTI-PASS: BISECT: running pass (5) EarlyCSEPass on function (f3)
; CHECK-LIMIT-MULTI-PASS: BISECT: NOT running pass (6) PostOrderFunctionAttrsPass on SCC (f1)
; CHECK-LIMIT-MULTI-PASS: BISECT: NOT running pass (7) EarlyCSEPass on function (f1)

declare i32 @g()

define void @f1() {
entry:
  br label %loop.0
loop.0:
  br i1 undef, label %loop.0.0, label %loop.1
loop.0.0:
  br i1 undef, label %loop.0.0, label %loop.0.1
loop.0.1:
  br i1 undef, label %loop.0.1, label %loop.0
loop.1:
  br i1 undef, label %loop.1, label %loop.1.bb1
loop.1.bb1:
  br i1 undef, label %loop.1, label %loop.1.bb2
loop.1.bb2:
  br i1 undef, label %end, label %loop.1.0
loop.1.0:
  br i1 undef, label %loop.1.0, label %loop.1
end:
  ret void
}

define i32 @f2() {
entry:
  ret i32 0
}

define i32 @f3() {
entry:
  %temp = call i32 @g()
  %icmp = icmp ugt i32 %temp, 2
  br i1 %icmp, label %bb.true, label %bb.false
bb.true:
  %temp2 = call i32 @f2()
  ret i32 %temp2
bb.false:
  ret i32 0
}
