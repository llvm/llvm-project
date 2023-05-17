; RUN: llc < %s

define void @f() {
entry:
  %0 = insertvalue { [0 x { ptr, ptr }], [0 x { ptr, i64 }] } undef, [0 x { ptr, ptr }] undef, 0
  ret void
}
