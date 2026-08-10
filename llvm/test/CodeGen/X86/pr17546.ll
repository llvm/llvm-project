; RUN: llc < %s -mtriple=x86_64-linux-gnu -mcpu=core-avx2 | FileCheck %s

define i32 @f_f___un_3C_unf_3E_un_3C_unf_3E_(<8 x i32> %__mask, i64 %BBBB) {
  %QQQ = trunc i64 %BBBB to i32
  %1 = extractelement <8 x i32> %__mask, i32 %QQQ
  ret i32 %1
}

; CHECK: f_f___un_3C_unf_3E_un_3C_unf_3E_
; CHECK: ret
