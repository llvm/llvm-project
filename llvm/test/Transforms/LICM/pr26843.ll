; RUN: opt -S -basicaa -licm < %s | FileCheck %s

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc18.0.0"

@v = common global i32 zeroinitializer, align 4 

; Make sure the store to v is not sunk past the memset
; CHECK-LABEL: @main
; CHECK: for.body:
; CHECK-NEXT: store i32 1, i32* @v
; CHECK-NEXT: tail call void @llvm.memset
; CHECK: end:
; CHECK-NEXT: ret i32 0

define i32 @main(i1 %k) {
entry:
  br label %for.body
 
for.body:
  store i32 1, i32* @v, align 4
  tail call void @llvm.memset.p0i8.i32(i8* bitcast (i32* @v to i8*), i8 0, i32 4, i32 4, i1 false)
  br label %for.latch
  
for.latch:
  br i1 %k, label %for.body, label %end

end:
  ret i32 0
}

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1)
