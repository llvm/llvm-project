; taken from X86 version of the same test
; RUN: llc -verify-machineinstrs -mtriple=powerpc-unknown-linux-gnu -O0 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -O0 < %s | FileCheck %s

@v1 = linkonce_odr local_unnamed_addr constant i32 32
; CHECK: .section  .rodata,"a"
; CHECK: .weak v1

define i32 @f1() {
  %x = load i32 , i32 * @v1
  ret i32 %x
}

@v2 = linkonce_odr constant i32 32
; CHECK: .weak v2

define i32* @f2() {
  ret i32* @v2
}

@v3 = linkonce_odr unnamed_addr constant i32 32
; CHECK: .section .rodata.cst4,"aM",
; CHECK: .weak v3

define i32* @f3() {
  ret i32* @v3
}

@v4 = linkonce_odr unnamed_addr global i32 32
; CHECK: .weak v4

define i32 @f4() {
  %x = load i32 , i32 * @v4
  ret i32 %x
}
