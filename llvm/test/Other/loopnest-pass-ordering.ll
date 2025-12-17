; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='no-op-loopnest' %s 2>&1 \
; RUN:     | FileCheck %s

;            @f()
;           /    \
;       loop.0   loop.1
;      /      \        \
; loop.0.0  loop.0.1  loop.1.0
;
; CHECK: Running pass: NoOpLoopNestPass on loop %loop.0 in function f
; CHECK: Running pass: NoOpLoopNestPass on loop %loop.1 in function f
; CHECK-NOT: Running pass: NoOpLoopNestPass on {{loop\..*\..*}}

define void @f(i1 %arg) {
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
