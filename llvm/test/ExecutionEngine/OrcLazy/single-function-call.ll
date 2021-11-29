; RUN: lli -jit-kind=orc-lazy %s
;
; Basic correctness check: We can make a call inside lazily JIT'd code.
; Compared to minimal.ll, this demonstrates that we can call through a stub.

define i32 @foo() {
entry:
  ret i32 0
}

define i32 @main(i32 %argc, i8** nocapture readnone %argv) {
entry:
  %0 = call i32() @foo()
  ret i32 %0
}
