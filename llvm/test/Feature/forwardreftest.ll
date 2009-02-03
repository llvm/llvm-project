; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

%myty = type i32 
%myfn = type float (i32,double,i32,i16)
type i32(%myfn*)
type i32(i32)
type i32(i32(i32)*)

  %thisfuncty = type i32 (i32) *

declare void @F(%thisfuncty, %thisfuncty, %thisfuncty)

define i32 @zarro(i32 %Func) {
Startup:
        add i32 0, 10           ; <i32>:0 [#uses=0]
        ret i32 0
}

define i32 @test(i32) {
        call void @F( %thisfuncty @zarro, %thisfuncty @test, %thisfuncty @foozball )
        ret i32 0
}

define i32 @foozball(i32) {
        ret i32 0
}

