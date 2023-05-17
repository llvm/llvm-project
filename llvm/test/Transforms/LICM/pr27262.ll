; RUN: opt -S -passes=licm < %s | FileCheck %s

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc18.0.0"

; Make sure the store to v is not sunk past the memset
; CHECK-LABEL: @main
; CHECK: for.body:
; CHECK-NEXT: store i8 1, ptr %p
; CHECK-NEXT: store i8 2, ptr %p1
; CHECK-NEXT: call void @llvm.memset
; CHECK: end:
; CHECK-NEXT: ret i32 0

define i32 @main(i1 %k, ptr %p) {
entry:
  %p1 = getelementptr i8, ptr %p, i32 1
  br label %for.body

for.body:
  store i8 1, ptr %p, align 1
  store i8 2, ptr %p1, align 1
  call void @llvm.memset.p0.i32(ptr %p, i8 255, i32 4, i1 false)
  br label %for.latch

for.latch:
  br i1 %k, label %for.body, label %end

end:
  ret i32 0
}

declare void @llvm.memset.p0.i32(ptr nocapture, i8, i32, i1)
