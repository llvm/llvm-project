; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: llvm-as -use-constant-ptrnull-for-fixed-length-splat=false -use-constant-ptrnull-for-scalable-splat=false < %s | llvm-dis -use-constant-ptrnull-for-fixed-length-splat=false -use-constant-ptrnull-for-scalable-splat=false | FileCheck %s --check-prefix=DISABLED
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
; CHECK: store [0 x i32] poison, ptr %x
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

; CHECK: @fixed_ptr_null_splat
; CHECK: ret <2 x ptr> splat (ptr null)
; DISABLED: @fixed_ptr_null_splat
; DISABLED: ret <2 x ptr> zeroinitializer
define <2 x ptr> @fixed_ptr_null_splat() {
  ret <2 x ptr> zeroinitializer
}

; CHECK: @scalable_ptr_null_splat
; CHECK: ret <vscale x 2 x ptr> splat (ptr null)
; DISABLED: @scalable_ptr_null_splat
; DISABLED: ret <vscale x 2 x ptr> zeroinitializer
define <vscale x 2 x ptr> @scalable_ptr_null_splat() {
  ret <vscale x 2 x ptr> zeroinitializer
}

; CHECK: @explicit_fixed_ptr_null_splat
; CHECK: ret <2 x ptr> splat (ptr null)
; DISABLED: @explicit_fixed_ptr_null_splat
; DISABLED: ret <2 x ptr> zeroinitializer
define <2 x ptr> @explicit_fixed_ptr_null_splat() {
  ret <2 x ptr> splat (ptr null)
}
