; RUN: llvm-as < %s | opt -instcombine | llvm-dis > %t
; RUN: grep {xor} %t
; RUN: grep {and} %t
; RUN: not grep {div} %t

define i1 @foo1(i1 %a, i1 %b) {
  %A = sub i1 %a, %b
  ret i1 %A
}

define i1 @foo2(i1 %a, i1 %b) {
  %A = mul i1 %a, %b
  ret i1 %A
}

define i1 @foo3(i1 %a, i1 %b) {
  %A = udiv i1 %a, %b
  ret i1 %A
}

define i1 @foo4(i1 %a, i1 %b) {
  %A = sdiv i1 %a, %b
  ret i1 %A
}
