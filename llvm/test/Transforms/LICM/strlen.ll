; RUN: opt -S -inferattrs -basicaa -licm < %s | FileCheck %s

define void @test(i64* noalias %loc, i8* noalias %a) {
; CHECK-LABEL: @test
; CHECK: @strlen
; CHECK-LABEL: loop:
  br label %loop

loop:
  %res = call i64 @strlen(i8* %a)
  store i64 %res, i64* %loc
  br label %loop
}

; CHECK: declare i64 @strlen(i8* nocapture) #0
; CHECK: attributes #0 = { argmemonly nounwind readonly }
declare i64 @strlen(i8*)


