; RUN: opt -p always-inline -enable-always-inliner-mem2reg -S -pass-remarks=inline < %s 2>&1 | FileCheck %s

; CHECK: remark: <unknown>:0:0: 'inner' inlined into 'middle' with (cost=always):
; CHECK: remark: <unknown>:0:0: 'inner' inlined into 'middle' with (cost=always):
; CHECK: remark: <unknown>:0:0: 'inner' inlined into 'middle' with (cost=always):
; CHECK-NEXT: remark: <unknown>:0:0: Promoting 1 allocas to SSA registers in function 'middle'
; CHECK-NEXT: remark: <unknown>:0:0: 'middle' inlined into 'outer' with (cost=always):

; This test constructs a call graph with no always-inline leaf functions,
; showing that the function with the most incoming always-inline calls
; is inlined first.

declare void @side(i32)

define linkonce_odr void @inner(i32* %x) #0 {
entry:
  store i32 42, i32* %x
  %v = load i32, i32* %x
  call void @side(i32 %v)
  call void @outer() ; not a always-inline leaf.
  ret void
}

define linkonce_odr void @middle() #0 {
entry:
  %var = alloca i32
  call void @inner(i32* %var)
  call void @inner(i32* %var)
  call void @inner(i32* %var)
  ret void
}

define void @outer() {
; CHECK-LABEL: outer()
; CHECK:       call void @side(i32 42)
; CHECK-NEXT:  call void @outer()
; CHECK-NEXT:  call void @side(i32 42)
; CHECK-NEXT:  call void @outer()
; CHECK-NEXT:  call void @side(i32 42)
; CHECK-NEXT:  call void @outer()
entry:
  call void @middle()
  ret void
}

attributes #0 = { alwaysinline }