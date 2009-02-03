; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll


declare i31 @"bar"(i31 %in) 

define i31 @"foo"(i31 %blah)
begin
  %xx = call i31 @bar(i31 %blah)
  ret i31 %xx
end

