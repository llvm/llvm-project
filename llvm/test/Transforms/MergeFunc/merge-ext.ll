; RUN: opt -S -passes=mergefunc < %s | FileCheck %s

@llvm.compiler.used = appending global [2 x ptr] [ptr @zext_a, ptr @sext_a], section "llvm.metadata"

define internal i32 @sext_a(i16 %a) unnamed_addr {
  %b = sext i16 %a to i32
  ret i32 %b
}

define i32 @sext_b(i16 %a) unnamed_addr {
  %b = sext i16 %a to i32
  ret i32 %b
}

define i32 @sext_c(i32 %a) unnamed_addr {
  %b = tail call i32 @sext_a(i32 %a)
  ret i32 %b
}

define internal i32 @zext_a(i16 %a) unnamed_addr {
  %b = zext i16 %a to i32
  ret i32 %b
}

define i32 @zext_b(i16 %a) unnamed_addr {
  %b = zext i16 %a to i32
  ret i32 %b
}

define i32 @zext_c(i32 %a) unnamed_addr {
  %b = tail call i32 @zext_a(i32 %a)
  ret i32 %b
}

; CHECK-LABEL: @llvm.compiler.used = appending global [2 x ptr] [ptr @zext_a, ptr @sext_a], section "llvm.metadata"

; CHECK-LABEL: define i32 @sext_b(i16 %a) unnamed_addr
; CHECK-NEXT:    sext
; CHECK-NEXT:    ret

; CHECK-LABEL: define i32 @sext_c(i32 %a) unnamed_addr
; CHECK-NEXT:    tail call i32 @sext_b(i32 %a)
; CHECK-NEXT:    ret

; CHECK-LABEL: define i32 @zext_b(i16 %a) unnamed_addr
; CHECK-NEXT:    zext
; CHECK-NEXT:    ret

; CHECK-LABEL: define i32 @zext_c(i32 %a) unnamed_addr
; CHECK-NEXT:    tail call i32 @zext_b(i32 %a)
; CHECK-NEXT:    ret

; CHECK-LABEL: define internal i32 @sext_a(i16 %0) unnamed_addr
; CHECK-NEXT:    tail call i32 @sext_b(i16 %0)
; CHECK-NEXT:    ret

; CHECK-LABEL: define internal i32 @zext_a(i16 %0) unnamed_addr
; CHECK-NEXT:    tail call i32 @zext_b(i16 %0)
; CHECK-NEXT:    ret
