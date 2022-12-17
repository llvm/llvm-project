; RUN: lli -jit-kind=mcjit -extra-module %p/Inputs/weak-function-2.ll %s
; RUN: lli -extra-module %p/Inputs/weak-function-2.ll %s
; UNSUPPORTED: uses_COFF
;
; Check that functions in two different modules agree on the address of weak
; function 'baz'
; Testing on COFF platforms is disabled as COFF has no representation of 'weak'
; linkage.

define weak i32 @baz() {
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

