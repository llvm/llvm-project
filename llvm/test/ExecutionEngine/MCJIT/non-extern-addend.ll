; RUN: %lli %s > /dev/null

define i32 @foo(i32 %x, i32 %y, double %d) {
entry:
  %d.int64 = bitcast double %d to i64
  %d.top64 = lshr i64 %d.int64, 32
  %d.top   = trunc i64 %d.top64 to i32
  %d.bottom = trunc i64 %d.int64 to i32
  %topCorrect = icmp eq i32 %d.top, 3735928559
  %bottomCorrect = icmp eq i32 %d.bottom, 4277009102
  %right = and i1 %topCorrect, %bottomCorrect
  %nRight = xor i1 %right, true
  %retVal = zext i1 %nRight to i32
  ret i32 %retVal
}

define i32 @main() {
entry:
  %call = call i32 @foo(i32 0, i32 1, double 0xDEADBEEFFEEDFACE)
  ret i32 %call
}
