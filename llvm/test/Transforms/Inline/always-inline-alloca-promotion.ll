; RUN: opt -p always-inline -enable-always-inliner-mem2reg -S -pass-remarks=inline < %s 2>&1 | FileCheck %s
; CHECK: remark: <unknown>:0:0: 'inner' inlined into 'middle' with (cost=always): always inline attribute
; CHECK-NEXT: remark: <unknown>:0:0: Promoting 1 allocas to SSA registers in function 'middle'
; CHECK-NEXT: remark: <unknown>:0:0: 'middle' inlined into 'outer' with (cost=always): always inline attribute

; A simple example to ensure we inline leaf function @inner into @middle,
; promote the alloca in @middle, then inline @middle into @outer.

declare void @side(i32)

define linkonce_odr void @inner(i32* %x) #0 {
entry:
  store i32 42, i32* %x
  %v = load i32, i32* %x
  call void @side(i32 %v)
  ret void
}

define linkonce_odr void @middle() #0 {
entry:
  %var = alloca i32
  call void @inner(i32* %var)
  ret void
}

define void @outer() {
; CHECK-LABEL: outer()
; CHECK:  call void @side(i32 42)
; CHECK:  ret void
entry:
  call void @middle()
  ret void
}

attributes #0 = { alwaysinline }