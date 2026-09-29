; RUN: llc -regalloc=pbqp < %s

define i32 @foo() {
entry:
  %call = tail call i32 (...) @baz()
  %call1 = tail call i32 (...) @baz()
  %call2 = tail call i32 (...) @baz()
  %call3 = tail call i32 (...) @baz()
  %call4 = tail call i32 (...) @baz()
  %call5 = tail call i32 (...) @baz()
  %call6 = tail call i32 (...) @baz()
  %call7 = tail call i32 (...) @baz()
  %call8 = tail call i32 (...) @baz()
  %call9 = tail call i32 (...) @baz()
  %call10 = tail call i32 (...) @baz()
  %call11 = tail call i32 (...) @baz()
  %call12 = tail call i32 (...) @baz()
  %call13 = tail call i32 (...) @baz()
  %call14 = tail call i32 (...) @baz()
  %call15 = tail call i32 (...) @baz()
  %call16 = tail call i32 (...) @baz()
  %call17 = tail call i32 @bar(i32 %call, i32 %call1, i32 %call2, i32 %call3, i32 %call4, i32 %call5, i32 %call6, i32 %call7, i32 %call8, i32 %call9, i32 %call10, i32 %call11, i32 %call12, i32 %call13, i32 %call14, i32 %call15, i32 %call16)
  ret i32 %call17
}

declare i32 @baz(...)

declare i32 @bar(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)

