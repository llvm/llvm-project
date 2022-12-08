; RUN: opt -S -passes='inferattrs,function(loop-mssa(licm))' < %s | FileCheck %s

define void @test(ptr noalias %loc, ptr noalias %a) {
; CHECK-LABEL: @test
; CHECK: @strlen
; CHECK-LABEL: loop:
  br label %loop

loop:
  %res = call i64 @strlen(ptr %a)
  store i64 %res, ptr %loc
  br label %loop
}

; CHECK: declare i64 @strlen(ptr nocapture) #0
; CHECK: attributes #0 = { mustprogress nofree nounwind willreturn memory(argmem: read) }
declare i64 @strlen(ptr)


