; RUN: llvm-as < %s | llvm-dis | grep 7 | count 3

define void @foo({i32, i32}* %x) nounwind {
  store {i32, i32}{i32 7, i32 9}, {i32, i32}* %x
  ret void
}
define void @foo_empty({}* %x) nounwind {
  store {}{}, {}* %x
  ret void
}
define void @bar([2 x i32]* %x) nounwind {
  store [2 x i32][i32 7, i32 9], [2 x i32]* %x
  ret void
}
define void @bar_empty([0 x i32]* %x) nounwind {
  store [0 x i32][], [0 x i32]* %x
  ret void
}
define void @qux(<{i32, i32}>* %x) nounwind {
  store <{i32, i32}><{i32 7, i32 9}>, <{i32, i32}>* %x
  ret void
}
define void @qux_empty(<{}>* %x) nounwind {
  store <{}><{}>, <{}>* %x
  ret void
}

