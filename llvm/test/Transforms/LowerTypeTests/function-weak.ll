; RUN: opt -S -passes=lowertypetests -mtriple=i686-unknown-linux-gnu %s | FileCheck --check-prefixes=CHECK,X86 %s
; RUN: opt -S -passes=lowertypetests -mtriple=x86_64-unknown-linux-gnu %s | FileCheck --check-prefixes=CHECK,X86 %s
; RUN: opt -S -passes=lowertypetests -mtriple=arm-unknown-linux-gnu %s | FileCheck --check-prefixes=CHECK,ARM %s
; RUN: opt -S -passes=lowertypetests -mtriple=aarch64-unknown-linux-gnu %s | FileCheck --check-prefixes=CHECK,ARM %s
; RUN: opt -S -passes=lowertypetests -mtriple=riscv32-unknown-linux-gnu %s | FileCheck --check-prefixes=CHECK,RISCV %s
; RUN: opt -S -passes=lowertypetests -mtriple=riscv64-unknown-linux-gnu %s | FileCheck --check-prefixes=CHECK,RISCV %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @x = global ptr null, align 8
@x = global ptr @f, align 8

; CHECK: @x2 = global ptr null, align 8
@x2 = global ptr @f, align 8

; CHECK: @x3 = internal global ptr null, align 8
@x3 = internal constant ptr @f, align 8

; f + addend
; CHECK: @x4 = global ptr null, align 8
@x4 = global ptr getelementptr (i8, ptr @f, i64 42), align 8

; aggregate initializer
; CHECK: @s = global { ptr, ptr, i32 } zeroinitializer, align 8
@s = global { ptr, ptr, i32 } { ptr @f, ptr @f, i32 42 }, align 8

; CHECK:  @llvm.global_ctors = appending global {{.*}}{ i32 0, ptr @__cfi_global_var_init

; CHECK: declare !type !0 extern_weak void @f()
declare !type !0 extern_weak void @f()

; CHECK: define zeroext i1 @check_f()
define zeroext i1 @check_f() {
entry:
; CHECK: ret i1 icmp ne (ptr select (i1 icmp ne (ptr @f, ptr null), ptr @[[JT:.*]], ptr null), ptr null)
  ret i1 icmp ne (ptr @f, ptr null)
}

; CHECK: define void @call_f() {
define void @call_f() {
entry:
; CHECK: call void @f()
  call void @f()
  ret void
}

declare i1 @llvm.type.test(ptr %ptr, metadata %bitset) nounwind readnone

define i1 @foo(ptr %p) {
  %x = call i1 @llvm.type.test(ptr %p, metadata !"typeid1")
  ret i1 %x
}

; X86: define private void @[[JT]]() #{{.*}} align 8 {
; ARM: define private void @[[JT]]() #{{.*}} align 4 {
; RISCV: define private void @[[JT]]() #{{.*}} align 8 {

; CHECK: define internal void @__cfi_global_var_init() section ".text.startup" {
; CHECK-NEXT: entry:
; CHECK-NEXT: store ptr select (i1 icmp ne (ptr @f, ptr null), ptr @[[JT]], ptr null), ptr @x, align 8
; CHECK-NEXT: store ptr select (i1 icmp ne (ptr @f, ptr null), ptr @[[JT]], ptr null), ptr @x2, align 8
; CHECK-NEXT: store ptr select (i1 icmp ne (ptr @f, ptr null), ptr @[[JT]], ptr null), ptr @x3, align 8
; CHECK-NEXT: store ptr getelementptr (i8, ptr select (i1 icmp ne (ptr @f, ptr null), ptr @[[JT]], ptr null), i64 42), ptr @x4, align 8
; CHECK-NEXT: store { ptr, ptr, i32 } { ptr select (i1 icmp ne (ptr @f, ptr null), ptr @[[JT]], ptr null), ptr select (i1 icmp ne (ptr @f, ptr null), ptr @[[JT]], ptr null), i32 42 }, ptr @s, align 8
; CHECK-NEXT: ret void
; CHECK-NEXT: }

!0 = !{i32 0, !"typeid1"}
