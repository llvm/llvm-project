; RUN: llc < %s -frame-pointer=all -mtriple=arm64-windows | FileCheck %s

; Test generated from C code:
; #include <stdarg.h>
; void *foo() {
;   return _AddressOfReturnAddress();
; }
; int bar(int x(va_list, void*), ...) {
;   va_list y;
;   va_start(y, x);
;   return x(y, _AddressOfReturnAddress()) + 1;
; }

declare void @llvm.va_start(i8*)
declare i8* @llvm.addressofreturnaddress()

define dso_local i8* @"foo"() {
entry:
  %0 = call i8* @llvm.addressofreturnaddress()
  ret i8* %0

; CHECK-LABEL: foo
; CHECK: stp x29, x30, [sp, #-16]!
; CHECK: mov x29, sp
; CHECK: add x0, x29, #8
; CHECK: ldp x29, x30, [sp], #16
}

define dso_local i32 @"bar"(i32 (i8*, i8*)* %x, ...) {
entry:
  %x.addr = alloca i32 (i8*, i8*)*, align 8
  %y = alloca i8*, align 8
  store i32 (i8*, i8*)* %x, i32 (i8*, i8*)** %x.addr, align 8
  %y1 = bitcast i8** %y to i8*
  call void @llvm.va_start(i8* %y1)
  %0 = load i32 (i8*, i8*)*, i32 (i8*, i8*)** %x.addr, align 8
  %1 = call i8* @llvm.addressofreturnaddress()
  %2 = load i8*, i8** %y, align 8
  %call = call i32 %0(i8* %2, i8* %1)
  %add = add nsw i32 %call, 1
  ret i32 %add

; CHECK-LABEL: bar
; CHECK: sub sp, sp, #96
; CHECK: stp x29, x30, [sp, #16]
; CHECK: add x29, sp, #16
; CHECK: str x1, [x29, #24]
; CHECK: add x1, x29, #8
; CHECK: ldp x29, x30, [sp, #16]
; CHECK: add sp, sp, #96
}
