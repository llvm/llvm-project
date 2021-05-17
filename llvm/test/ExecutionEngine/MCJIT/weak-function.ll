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

define i8* @foo() {
entry:
  ret i8* bitcast (i32 ()* @baz to i8*)
}

declare i8* @bar()

define i32 @main(i32 %argc, i8** %argv) {
entry:
  %call = tail call i8* @foo()
  %call1 = tail call i8* @bar()
  %cmp = icmp ne i8* %call, %call1
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

