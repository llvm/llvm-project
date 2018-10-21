; RUN: lli -jit-kind=orc-lazy %s
;
; Basic sanity check: A module with a single no-op main function runs.

define i32 @main(i32 %argc, i8** nocapture readnone %argv) {
entry:
  ret i32 0
}
