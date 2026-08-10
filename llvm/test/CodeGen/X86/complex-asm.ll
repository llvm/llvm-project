; RUN: llc < %s -mtriple=x86_64-apple-darwin
; This formerly crashed.

%0 = type { i64, i64 }

define %0 @f() nounwind ssp {
entry:
  %v = alloca %0, align 8
  call void asm sideeffect "", "=*r,r,r,0,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(%0) %v, i32 0, i32 1, i128 undef) nounwind
  %0 = getelementptr inbounds %0, ptr %v, i64 0, i32 0
  %1 = load i64, ptr %0, align 8
  %2 = getelementptr inbounds %0, ptr %v, i64 0, i32 1
  %3 = load i64, ptr %2, align 8
  %mrv4 = insertvalue %0 undef, i64 %1, 0
  %mrv5 = insertvalue %0 %mrv4, i64 %3, 1
  ret %0 %mrv5
}
