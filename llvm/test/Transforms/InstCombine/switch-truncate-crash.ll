; RUN: opt -instcombine < %s

define void @test() {
  switch i32 0, label %out [i32 0, label %out]
out:
  ret void
}
