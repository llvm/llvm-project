; RUN: opt -passes=alignment-from-assumptions -disable-output < %s

; REQUIRES: asserts

; The alignment assumption is a global, which has users in a different
; function. Test that in this case the dominator tree is only queried with
; blocks from the same function.

target triple = "x86_64-unknown-linux-gnu"

@global = external constant [192 x i8]

define void @fn1() {
  call void @llvm.assume(i1 false) [ "align"(ptr @global, i64 1) ]
  ret void
}

define void @fn2() {
  ret void

loop:
  %gep = getelementptr inbounds i8, ptr @global, i64 0
  %load = load i64, ptr %gep, align 1
  br label %loop
}
