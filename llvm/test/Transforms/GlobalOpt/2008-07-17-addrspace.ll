; This test lets globalopt split the global struct and array into different
; values. This used to crash, because globalopt forgot to put the new var in the
; same address space as the old one.

; RUN: opt < %s -passes=globalopt -S | FileCheck %s

; Check that the new global values still have their address space
; CHECK: addrspace(1) global
; CHECK: addrspace(1) global

@struct = internal addrspace(1) global { i32, i32 } zeroinitializer
@array = internal addrspace(1) global [ 2 x i32 ] zeroinitializer 

define i32 @foo() {
  %A = load i32, ptr addrspace(1) getelementptr ({ i32, i32 }, ptr addrspace(1) @struct, i32 0, i32 0)
  %B = load i32, ptr addrspace(1) @array
  ; Use the loaded values, so they won't get removed completely
  %R = add i32 %A, %B
  ret i32 %R
}

; We put stores in a different function, so that the global variables won't get
; optimized away completely.
define void @bar(i32 %R) {
  store i32 %R, ptr addrspace(1) @array
  store i32 %R, ptr addrspace(1) getelementptr ({ i32, i32 }, ptr addrspace(1) @struct, i32 0, i32 0)
  ret void
}
