; RUN: opt < %s -instcombine -S | grep "store i32 0,"
; PR4366

define void @a() {
  store i32 0, i32 addrspace(1)* null
  ret void
}
