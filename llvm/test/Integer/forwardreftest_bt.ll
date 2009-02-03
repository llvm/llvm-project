; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

  %myty = type i55 
  %myfn = type float (i55,double,i55,i16)
  type i55(%myfn*)
  type i55(i55)
  type i55(i55(i55)*)

  %thisfuncty = type i55 (i55) *

declare void @F(%thisfuncty, %thisfuncty, %thisfuncty)

; This function always returns zero
define i55 @zarro(i55 %Func)
begin
Startup:
    add i55 0, 10
    ret i55 0 
end

define i55 @test(i55) 
begin
    call void @F(%thisfuncty @zarro, %thisfuncty @test, %thisfuncty @foozball)
    ret i55 0
end

define i55 @foozball(i55)
begin
    ret i55 0
end

