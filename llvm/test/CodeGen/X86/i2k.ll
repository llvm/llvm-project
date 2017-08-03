; RUN: llc < %s -march=x86

define void @foo(i2011* %x, i2011* %y, i2011* %p) nounwind {
  %a = load i2011, i2011* %x
  %b = load i2011, i2011* %y
  %c = add i2011 %a, %b
  store i2011 %c, i2011* %p
  ret void
}
