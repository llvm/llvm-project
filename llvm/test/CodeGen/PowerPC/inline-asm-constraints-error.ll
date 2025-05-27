; RUN: not llc -mcpu=pwr8 -mtriple=powerpc64-ibm-aix-xcoff  -filetype=null %s 2>&1 | FileCheck %s
; XFAIL: *

; CHECK: LLVM ERROR: Do not know how to promote this operator's operand!
define i8 @charLoadAConstrained(i8 zeroext %ptr, i32 %index) {
entry:
  %0 = tail call i16 asm "lbzx $0,$1,$2", "=r,a,r"(i8 %ptr, i32 %index)
  %conv = trunc i16 %0 to i8
  ret i8 %conv
}

define i8 @shortLoadAConstrained(i16 zeroext %ptr, i32 %index) {
; CHECK: LLVM ERROR: Do not know how to promote this operator's operand!
entry:
  %0 = tail call i16 asm "lbzx $0,$1,$2", "=r,a,r"(i16 %ptr, i32 %index)
  %conv = trunc i16 %0 to i8
  ret i8 %conv
}

define i8 @intLoadAConstrained(i32 zeroext %ptr, i32 %index) {
; CHECK: LLVM ERROR: Do not know how to promote this operator's operand!
entry:
  %0 = tail call i16 asm "lbzx $0,$1,$2", "=r,a,r"(i32 %ptr, i32 %index)
  %conv = trunc i16 %0 to i8
  ret i8 %conv
}
