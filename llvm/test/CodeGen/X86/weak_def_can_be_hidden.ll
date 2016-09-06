; RUN: llc -mtriple=x86_64-apple-darwin11 -O0 < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-apple-darwin10 -O0 < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-apple-darwin9 -O0 < %s | FileCheck --check-prefix=CHECK-D89 %s
; RUN: llc -mtriple=i686-apple-darwin9 -O0 < %s | FileCheck --check-prefix=CHECK-D89 %s
; RUN: llc -mtriple=i686-apple-darwin8 -O0 < %s | FileCheck --check-prefix=CHECK-D89 %s

@v1 = linkonce_odr local_unnamed_addr constant i32 32
; CHECK: .globl  _v1
; CHECK: .weak_def_can_be_hidden _v1

; CHECK-D89: .globl  _v1
; CHECK-D89: .weak_definition _v1

define i32 @f1() {
  %x = load i32 , i32 * @v1
  ret i32 %x
}

@v2 = linkonce_odr constant i32 32
; CHECK: .globl  _v2
; CHECK: .weak_definition _v2

; CHECK-D89: .globl  _v2
; CHECK-D89: .weak_definition _v2

define i32* @f2() {
  ret i32* @v2
}

@v3 = linkonce_odr unnamed_addr constant i32 32
; CHECK: .globl  _v3
; CHECK: .weak_def_can_be_hidden _v3

; CHECK-D89: .globl  _v3
; CHECK-D89: .weak_definition _v3

define i32* @f3() {
  ret i32* @v3
}

@v4 = linkonce_odr unnamed_addr global i32 32
; CHECK: .globl  _v4
; CHECK: .weak_def_can_be_hidden _v4

; CHECK-D89: .globl  _v4
; CHECK-D89: .weak_definition _v4

define i32 @f4() {
  %x = load i32 , i32 * @v4
  ret i32 %x
}
