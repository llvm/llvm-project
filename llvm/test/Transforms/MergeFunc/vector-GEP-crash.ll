; RUN: opt -passes=mergefunc -disable-output < %s
; This used to cause a crash when compairing the GEPs

define void @foo(<2 x ptr>) {
  %tmp = getelementptr i64, <2 x ptr> %0, <2 x i64> <i64 0, i64 0>
  ret void
}

define void @bar(<2 x ptr>) {
  %tmp = getelementptr i64, <2 x ptr> %0, <2 x i64> <i64 0, i64 0>
  ret void
}
