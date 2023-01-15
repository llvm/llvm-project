; RUN: llc < %s -frame-pointer=all -mtriple=arm64-windows | FileCheck %s

; Test generated from C code:
; #include <stdarg.h>
; ptr foo() {
;   return _AddressOfReturnAddress();
; }
; int bar(int x(va_list, ptr), ...) {
;   va_list y;
;   va_start(y, x);
;   return x(y, _AddressOfReturnAddress()) + 1;
; }

declare void @llvm.va_start(ptr)
declare ptr @llvm.addressofreturnaddress()

define dso_local ptr @"foo"() {
entry:
  %0 = call ptr @llvm.addressofreturnaddress()
  ret ptr %0

; CHECK-LABEL: foo
; CHECK: stp x29, x30, [sp, #-16]!
; CHECK: mov x29, sp
; CHECK: add x0, x29, #8
; CHECK: ldp x29, x30, [sp], #16
}

define dso_local i32 @"bar"(ptr %x, ...) {
entry:
  %x.addr = alloca ptr, align 8
  %y = alloca ptr, align 8
  store ptr %x, ptr %x.addr, align 8
  call void @llvm.va_start(ptr %y)
  %0 = load ptr, ptr %x.addr, align 8
  %1 = call ptr @llvm.addressofreturnaddress()
  %2 = load ptr, ptr %y, align 8
  %call = call i32 %0(ptr %2, ptr %1)
  %add = add nsw i32 %call, 1
  ret i32 %add

; CHECK-LABEL: bar
; CHECK: sub sp, sp, #96
; CHECK: stp x29, x30, [sp, #16]
; CHECK: add x29, sp, #16
; CHECK: stp x1, x2, [x29, #24]
; CHECK: add x1, x29, #8
; CHECK: ldp x29, x30, [sp, #16]
; CHECK: add sp, sp, #96
}
