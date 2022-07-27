; RUN: llc < %s -mtriple=i686--

define void @foo(ptr %x, ptr %y, ptr %p) nounwind {
  %a = load i2011, ptr %x
  %b = load i2011, ptr %y
  %c = add i2011 %a, %b
  store i2011 %c, ptr %p
  ret void
}
