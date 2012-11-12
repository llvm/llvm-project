; This test lets globalopt split the global struct and array into different
; values. This used to crash, because globalopt forgot to put the new var in the
; same address space as the old one.

; RUN: opt < %s -globalopt -S > %t
; Check that the new global values still have their address space
; RUN: cat %t | grep addrspace.*global

@struct = internal addrspace(1) global { i32, i32 } zeroinitializer
@array = internal addrspace(1) global [ 2 x i32 ] zeroinitializer 

define i32 @foo() {
  %A = load i32 addrspace(1) * getelementptr ({ i32, i32 } addrspace(1) * @struct, i32 0, i32 0)
  %B = load i32 addrspace(1) * getelementptr ([ 2 x i32 ] addrspace(1) * @array, i32 0, i32 0)
  ; Use the loaded values, so they won't get removed completely
  %R = add i32 %A, %B
  ret i32 %R
}

; We put stores in a different function, so that the global variables won't get
; optimized away completely.
define void @bar(i32 %R) {
  store i32 %R, i32 addrspace(1) * getelementptr ([ 2 x i32 ] addrspace(1) * @array, i32 0, i32 0)
  store i32 %R, i32 addrspace(1) * getelementptr ({ i32, i32 } addrspace(1) * @struct, i32 0, i32 0)
  ret void
}


