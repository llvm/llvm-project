; This test uses the same IR functions of the opt-bisect test
; and it checks if it correctly cohexists with the opt-disable flag.
; Passes before opt-bisect-limit should be ran unless disabled 
; by opt-disable. Later passes NOT run.

; RUN: opt -disable-output -disable-verify \
; RUN:     -opt-disable=inferattrs,function-attrs  \
; RUN:     -opt-bisect-limit=6 \
; RUN:     -passes='inferattrs,cgscc(function-attrs,function(early-cse))' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-MULTI-PASS
; CHECK-MULTI-PASS: BISECT: NOT running pass (1) inferattrs on [module]
; CHECK-MULTI-PASS: BISECT: NOT running pass (2) function-attrs on (f1)
; CHECK-MULTI-PASS: BISECT: running pass (3) early-cse on f1
; CHECK-MULTI-PASS: BISECT: NOT running pass (4) function-attrs on (f2)
; CHECK-MULTI-PASS: BISECT: running pass (5) early-cse on f2
; CHECK-MULTI-PASS: BISECT: NOT running pass (6) function-attrs on (f3)
; CHECK-MULTI-PASS: BISECT: NOT running pass (7) early-cse on f3
; CHECK-MULTI-PASS: BISECT: NOT running pass (8) function-attrs on (f4)
; CHECK-MULTI-PASS: BISECT: NOT running pass (9) early-cse on f4

declare i32 @g()

define void @f1(i1 %arg) {
entry:
  br label %loop.0
loop.0:
  br i1 %arg, label %loop.0.0, label %loop.1
loop.0.0:
  br i1 %arg, label %loop.0.0, label %loop.0.1
loop.0.1:
  br i1 %arg, label %loop.0.1, label %loop.0
loop.1:
  br i1 %arg, label %loop.1, label %loop.1.bb1
loop.1.bb1:
  br i1 %arg, label %loop.1, label %loop.1.bb2
loop.1.bb2:
  br i1 %arg, label %end, label %loop.1.0
loop.1.0:
  br i1 %arg, label %loop.1.0, label %loop.1
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

define void @f4(i1 %arg) {
entry:
  %i = alloca i32, align 4
  call void @llvm.lifetime.start(i64 4, ptr %i)
  br label %for.cond

for.cond:
  br i1 %arg, label %for.body, label %for.end

for.body:
  br label %for.cond

for.end:
  ret void
}

declare void @llvm.lifetime.start(i64, ptr nocapture)
