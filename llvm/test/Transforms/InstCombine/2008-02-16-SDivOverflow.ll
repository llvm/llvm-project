; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {ret i.* 0} | count 2
; PR2048

define i32 @i(i32 %a) {
  %tmp1 = sdiv i32 %a, -1431655765
  %tmp2 = sdiv i32 %tmp1, 3
  ret i32 %tmp2
}

define i8 @j(i8 %a) {
  %tmp1 = sdiv i8 %a, 64
  %tmp2 = sdiv i8 %tmp1, 3
  ret i8 %tmp2
}
