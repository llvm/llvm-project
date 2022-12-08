; Can be used as an entry point for tests that rely purely on static
; initializer side-effects.

define i32 @main(i32 %argc, ptr %argv) {
entry:
  ret i32 0
}
