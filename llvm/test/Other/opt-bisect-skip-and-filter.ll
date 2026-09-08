; Tests that verify functionality for -opt-disable and -opt-bisect-funcs

; If enabling bisect function filtering, do not skip Module Passes.
; RUN: opt -disable-output -disable-verify -passes=inferattrs \
; RUN:     -opt-bisect-funcs=f1 -opt-bisect-limit=-1 -opt-bisect-verbose=true %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-MODULE-PASS-FILTER
; CHECK-MODULE-PASS-FILTER: BISECT: running pass (1) inferattrs on [module]
; CHECK-MODULE-PASS-FILTER-NOT: BISECT: NOT running

; RUN: opt -disable-output -disable-verify \
; RUN:     -passes=early-cse -opt-bisect-limit=-1 -opt-bisect-funcs=f1 -opt-bisect-verbose=true %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-FUNCTION-PASS-FILTER
; CHECK-FUNCTION-PASS-FILTER: BISECT: running pass (1) early-cse on f1
; CHECK-FUNCTION-PASS-FILTER: BISECT: NOT running pass (2) early-cse on f2
; CHECK-FUNCTION-PASS-FILTER: BISECT: NOT running pass (3) early-cse on f3
; CHECK-FUNCTION-PASS-FILTER: BISECT: NOT running pass (4) early-cse on f4

; RUN: opt -disable-output -disable-verify \
; RUN:     -passes=early-cse -opt-bisect-limit=2 -opt-bisect-funcs=f1,f2,f3 -opt-bisect-verbose=true %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-LIMIT-FUNCTION-PASS-FILTER
; CHECK-LIMIT-FUNCTION-PASS-FILTER: BISECT: running pass (1) early-cse on f1
; CHECK-LIMIT-FUNCTION-PASS-FILTER: BISECT: running pass (2) early-cse on f2
; CHECK-LIMIT-FUNCTION-PASS-FILTER: BISECT: NOT running pass (3) early-cse on f3
; CHECK-LIMIT-FUNCTION-PASS-FILTER: BISECT: NOT running pass (4) early-cse on f4

; RUN: opt -disable-output -disable-verify \
; RUN:     -passes=function-attrs -opt-bisect-limit=-1 -opt-bisect-funcs=f2 -opt-bisect-verbose=true %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-CGSCC-PASS-FILTER
; CHECK-CGSCC-PASS-FILTER: BISECT: running pass (1) function-attrs on (f1)
; CHECK-CGSCC-PASS-FILTER-NOT: BISECT: Skip bisecting pass 'function-attrs' on
; CHECK-CGSCC-PASS-FILTER: BISECT: running pass (2) function-attrs on (f2)
; CHECK-CGSCC-PASS-FILTER: BISECT: running pass (3) function-attrs on (f3)
; CHECK-CGSCC-PASS-FILTER: BISECT: running pass (4) function-attrs on (f4)


; RUN: opt -disable-output -disable-verify -opt-disable=3,7 \
; RUN:     -passes='inferattrs,cgscc(function-attrs,function(early-cse))' -opt-bisect-verbose=true %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-DISABLE-PASS
; CHECK-DISABLE-PASS: BISECT: running pass (1) inferattrs on [module]
; CHECK-DISABLE-PASS: BISECT: running pass (2) function-attrs on (f1)
; CHECK-DISABLE-PASS: BISECT: NOT running pass (3) early-cse on f1
; CHECK-DISABLE-PASS: BISECT: running pass (4) function-attrs on (f2)
; CHECK-DISABLE-PASS: BISECT: running pass (5) early-cse on f2
; CHECK-DISABLE-PASS: BISECT: running pass (6) function-attrs on (f3)
; CHECK-DISABLE-PASS: BISECT: NOT running pass (7) early-cse on f3
; CHECK-DISABLE-PASS: BISECT: running pass (8) function-attrs on (f4)
; CHECK-DISABLE-PASS: BISECT: running pass (9) early-cse on f4

declare i32 @g()

define void @f1(i1 %cond0, i1 %cond1, i1 %cond2, i1 %cond3, i1 %cond4,
                i1 %cond5, i1 %cond6) {
entry:
  br label %loop.0
loop.0:
  br i1 %cond0, label %loop.0.0, label %loop.1
loop.0.0:
  br i1 %cond1, label %loop.0.0, label %loop.0.1
loop.0.1:
  br i1 %cond2, label %loop.0.1, label %loop.0
loop.1:
  br i1 %cond3, label %loop.1, label %loop.1.bb1
loop.1.bb1:
  br i1 %cond4, label %loop.1, label %loop.1.bb2
loop.1.bb2:
  br i1 %cond5, label %end, label %loop.1.0
loop.1.0:
  br i1 %cond6, label %loop.1.0, label %loop.1
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

; This function is here to verify that opt-bisect can skip all passes for
; functions that contain lifetime intrinsics.
define void @f4(i1 %cond) {
entry:
  %i = alloca i32, align 4
  call void @llvm.lifetime.start(i64 4, ptr %i)
  br label %for.cond

for.cond:
  br i1 %cond, label %for.body, label %for.end

for.body:
  br label %for.cond

for.end:
  ret void
}

declare void @llvm.lifetime.start(i64, ptr nocapture)
