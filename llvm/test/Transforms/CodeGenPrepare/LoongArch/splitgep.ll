; RUN: opt --mtriple=loongarch64 -mattr=+d -S --passes='require<profile-summary>,function(codegenprepare)' %s | FileCheck %s

; Check that we have deterministic output
define void @test(ptr %sp, ptr %t, i32 %n) {
; CHECK-LABEL: @test(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %splitgep1 = getelementptr i8, ptr %t, i64 80000
; CHECK-NEXT:    %s = load ptr, ptr %sp
; CHECK-NEXT:    %splitgep = getelementptr i8, ptr %s, i64 80000
entry:
  %s = load ptr, ptr %sp
  br label %while_cond

while_cond:
  %phi = phi i32 [ 0, %entry ], [ %i, %while_body ]
  %gep0 = getelementptr [65536 x i32], ptr %s, i64 0, i64 20000
  %gep1 = getelementptr [65536 x i32], ptr %s, i64 0, i64 20001
  %gep2 = getelementptr [65536 x i32], ptr %t, i64 0, i64 20000
  %gep3 = getelementptr [65536 x i32], ptr %t, i64 0, i64 20001
  %cmp = icmp slt i32 %phi, %n
  br i1 %cmp, label %while_body, label %while_end

while_body:
  %i = add i32 %phi, 1
  store i32 %i, ptr %gep0
  store i32 %phi, ptr %gep1
  store i32 %i, ptr %gep2
  store i32 %phi, ptr %gep3
  br label %while_cond

while_end:
  ret void
}

