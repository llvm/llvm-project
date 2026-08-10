; RUN: opt < %s -passes=instcombine -S | grep "store i32 0,"
; PR4366

define void @a() {
  store i32 0, ptr addrspace(1) null
  ret void
}
