; RUN: lli -jit-kind=orc-lazy %s
;
; Basic correctness check: A module with a single no-op main function runs.

define i32 @main(i32 %argc, ptr nocapture readnone %argv) {
entry:
  ret i32 0
}
