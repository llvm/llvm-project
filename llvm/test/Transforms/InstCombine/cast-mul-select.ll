; RUN: llvm-as < %s | opt -instcombine | llvm-dis | notcast

define i32 @mul(i32 %x, i32 %y) {
  %A = trunc i32 %x to i8
  %B = trunc i32 %y to i8
  %C = mul i8 %A, %B
  %D = zext i8 %C to i32
  ret i32 %D
}

define i32 @select1(i1 %cond, i32 %x, i32 %y, i32 %z) {
  %A = trunc i32 %x to i8
  %B = trunc i32 %y to i8
  %C = trunc i32 %z to i8
  %D = add i8 %A, %B
  %E = select i1 %cond, i8 %C, i8 %D
  %F = zext i8 %E to i32
  ret i32 %F
}

define i8 @select2(i1 %cond, i8 %x, i8 %y, i8 %z) {
  %A = zext i8 %x to i32
  %B = zext i8 %y to i32
  %C = zext i8 %z to i32
  %D = add i32 %A, %B
  %E = select i1 %cond, i32 %C, i32 %D
  %F = trunc i32 %E to i8
  ret i8 %F
}
