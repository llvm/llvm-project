; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

define void @cond_call() {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %foo.ret = call i64 @foo()
  br label %for.body
}

declare i64 @foo() #0

attributes #0 = {"vector-function-abi-variant"="_ZGV_LLVM_M4v_foo(vector_foo)" }
; CHECK:      invalid name for a VFABI variant: _ZGV_LLVM_M4v_foo(vector_foo)
; CHECK-NEXT: ptr @foo
