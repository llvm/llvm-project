; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {sdiv i8 \%a, 9}
; PR2048

define i8 @i(i8 %a) {
  %tmp1 = sdiv i8 %a, -3
  %tmp2 = sdiv i8 %tmp1, -3
  ret i8 %tmp2
}

