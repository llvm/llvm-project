; RUN: llc -march=mips < %s | FileCheck %s

define i64 @add64(i64 %u, i64 %v) nounwind  {
entry:
; CHECK: addu
; CHECK: sltu 
; CHECK: addu
; CHECK: addu
  %tmp2 = add i64 %u, %v  
  ret i64 %tmp2
}

define i64 @sub64(i64 %u, i64 %v) nounwind  {
entry:
; CHECK: sub64
; CHECK: subu
; CHECK: sltu 
; CHECK: addu
; CHECK: subu
  %tmp2 = sub i64 %u, %v
  ret i64 %tmp2
}
