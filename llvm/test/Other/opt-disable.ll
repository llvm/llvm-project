; RUN: opt -disable-output -disable-verify \
; RUN:     -opt-disable-enable-verbosity \
; RUN:     -passes=inferattrs -opt-disable=inferfunctionattrspass %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-MODULE-PASS
; CHECK-MODULE-PASS: DISABLE: NOT running pass InferFunctionAttrsPass on [module]

; RUN: opt -disable-output -disable-verify \
; RUN:     -opt-disable-enable-verbosity \
; RUN:     -passes=sroa -opt-disable=sroapass %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-FUNCTION-PASS
; CHECK-FUNCTION-PASS: DISABLE: NOT running pass SROAPass on f1
; CHECK-FUNCTION-PASS: DISABLE: NOT running pass SROAPass on f2
; CHECK-FUNCTION-PASS: DISABLE: NOT running pass SROAPass on f3
; CHECK-FUNCTION-PASS: DISABLE: NOT running pass SROAPass on f4

; RUN: opt -disable-output -disable-verify \
; RUN:     -opt-disable=inferfunctionattrspass,PostOrderFunctionAttrsPass \
; RUN:     -opt-disable-enable-verbosity \
; RUN:     -passes='inferattrs,cgscc(function-attrs,function(early-cse))' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-MULTI-PASS
; CHECK-MULTI-PASS: DISABLE: NOT running pass InferFunctionAttrsPass on [module]
; CHECK-MULTI-PASS: DISABLE: NOT running pass PostOrderFunctionAttrsPass on (f1)
; CHECK-MULTI-PASS: DISABLE: running pass EarlyCSEPass on f1
; CHECK-MULTI-PASS: DISABLE: NOT running pass PostOrderFunctionAttrsPass on (f2)
; CHECK-MULTI-PASS: DISABLE: running pass EarlyCSEPass on f2
; CHECK-MULTI-PASS: DISABLE: NOT running pass PostOrderFunctionAttrsPass on (f3)
; CHECK-MULTI-PASS: DISABLE: running pass EarlyCSEPass on f3
; CHECK-MULTI-PASS: DISABLE: NOT running pass PostOrderFunctionAttrsPass on (f4)
; CHECK-MULTI-PASS: DISABLE: running pass EarlyCSEPass on f4

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

; This function is here to verify that opt-bisect can skip all passes for
; functions that contain lifetime intrinsics.
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