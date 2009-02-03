; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

%x = type i19


define i19 @"foo"(i19 %in) 
begin
label: 
  ret i19 2
end

