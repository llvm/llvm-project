; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

; CHECK: @foo
; CHECK: store { i32, i32 } { i32 7, i32 9 }, { i32, i32 }* %x
; CHECK: ret
define void @foo({i32, i32}* %x) nounwind {
  store {i32, i32}{i32 7, i32 9}, {i32, i32}* %x
  ret void
}

; CHECK: @foo_empty
; CHECK: store {} zeroinitializer, {}* %x
; CHECK: ret
define void @foo_empty({}* %x) nounwind {
  store {}{}, {}* %x
  ret void
}

; CHECK: @bar
; CHECK: store [2 x i32] [i32 7, i32 9], [2 x i32]* %x
; CHECK: ret
define void @bar([2 x i32]* %x) nounwind {
  store [2 x i32][i32 7, i32 9], [2 x i32]* %x
  ret void
}

; CHECK: @bar_empty
; CHECK: store [0 x i32] undef, [0 x i32]* %x
; CHECK: ret
define void @bar_empty([0 x i32]* %x) nounwind {
  store [0 x i32][], [0 x i32]* %x
  ret void
}

; CHECK: @qux
; CHECK: store <{ i32, i32 }> <{ i32 7, i32 9 }>, <{ i32, i32 }>* %x
; CHECK: ret
define void @qux(<{i32, i32}>* %x) nounwind {
  store <{i32, i32}><{i32 7, i32 9}>, <{i32, i32}>* %x
  ret void
}

; CHECK: @qux_empty
; CHECK: store <{}> zeroinitializer, <{}>* %x
; CHECK: ret
define void @qux_empty(<{}>* %x) nounwind {
  store <{}><{}>, <{}>* %x
  ret void
}

