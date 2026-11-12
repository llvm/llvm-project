; RUN: lli -jit-kind=orc-lazy -extra-module %p/Inputs/weak-function-2.ll %s
;
; Check that functions in two different modules agree on the address of weak
; function 'baz'

define linkonce_odr i32 @baz() {
entry:
  ret i32 0
}

define ptr @foo() {
entry:
  ret ptr @baz
}

declare ptr @bar()

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %call = tail call ptr @foo()
  %call1 = tail call ptr @bar()
  %cmp = icmp ne ptr %call, %call1
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

