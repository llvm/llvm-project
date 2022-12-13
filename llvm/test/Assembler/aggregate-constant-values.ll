; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: @foo
; CHECK: store { i32, i32 } { i32 7, i32 9 }, ptr %x
; CHECK: ret
define void @foo(ptr %x) nounwind {
  store {i32, i32}{i32 7, i32 9}, ptr %x
  ret void
}

; CHECK: @foo_empty
; CHECK: store {} zeroinitializer, ptr %x
; CHECK: ret
define void @foo_empty(ptr %x) nounwind {
  store {}{}, ptr %x
  ret void
}

; CHECK: @bar
; CHECK: store [2 x i32] [i32 7, i32 9], ptr %x
; CHECK: ret
define void @bar(ptr %x) nounwind {
  store [2 x i32][i32 7, i32 9], ptr %x
  ret void
}

; CHECK: @bar_empty
; CHECK: store [0 x i32] undef, ptr %x
; CHECK: ret
define void @bar_empty(ptr %x) nounwind {
  store [0 x i32][], ptr %x
  ret void
}

; CHECK: @qux
; CHECK: store <{ i32, i32 }> <{ i32 7, i32 9 }>, ptr %x
; CHECK: ret
define void @qux(ptr %x) nounwind {
  store <{i32, i32}><{i32 7, i32 9}>, ptr %x
  ret void
}

; CHECK: @qux_empty
; CHECK: store <{}> zeroinitializer, ptr %x
; CHECK: ret
define void @qux_empty(ptr %x) nounwind {
  store <{}><{}>, ptr %x
  ret void
}

