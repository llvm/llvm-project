; RUN: opt %s -passes=deadargelim -S | FileCheck %s


@block_addr = global ptr blockaddress(@varargs_func, %l1)
; CHECK: @block_addr = global ptr blockaddress(@varargs_func, %l1)


; This function is referenced by a "blockaddress" constant but it is
; not address-taken, so the pass should be able to remove its unused
; varargs.

define internal i32 @varargs_func(ptr %addr, ...) {
  indirectbr ptr %addr, [ label %l1, label %l2 ]
l1:
  ret i32 1
l2:
  ret i32 2
}
; CHECK: define internal i32 @varargs_func(ptr %addr) {

define i32 @caller(ptr %addr) {
  %r = call i32 (ptr, ...) @varargs_func(ptr %addr)
  ret i32 %r
}
; CHECK: %r = call i32 @varargs_func(ptr %addr)
