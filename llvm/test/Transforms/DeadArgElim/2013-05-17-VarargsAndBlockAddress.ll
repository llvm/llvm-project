; RUN: opt %s -passes=deadargelim -S | FileCheck %s


@block_addr = global i8* blockaddress(@varargs_func, %l1)
; CHECK: @block_addr = global i8* blockaddress(@varargs_func, %l1)


; This function is referenced by a "blockaddress" constant but it is
; not address-taken, so the pass should be able to remove its unused
; varargs.

define internal i32 @varargs_func(i8* %addr, ...) {
  indirectbr i8* %addr, [ label %l1, label %l2 ]
l1:
  ret i32 1
l2:
  ret i32 2
}
; CHECK: define internal i32 @varargs_func(i8* %addr) {

define i32 @caller(i8* %addr) {
  %r = call i32 (i8*, ...) @varargs_func(i8* %addr)
  ret i32 %r
}
; CHECK: %r = call i32 @varargs_func(i8* %addr)
