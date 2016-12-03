; ModuleID = 'thinlto-function-summary-callgraph-profile-summary2.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"


define void @hot1() #1 {
  ret void
}
define void @hot2() #1 {
  ret void
}
define void @hot3() #1 {
  ret void
}
define void @cold() #1 {
  ret void
}
define void @none1() #1 {
  ret void
}
define void @none2() #1 {
  ret void
}
define void @none3() #1 {
  ret void
}

