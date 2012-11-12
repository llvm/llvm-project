; RUN: llc < %s -march=x86 | grep "(%esp"
; RUN: llc < %s -march=x86 | grep "pushl	%ebp" | count 1
; RUN: llc < %s -march=x86 | grep "popl	%ebp" | count 1

declare i8* @llvm.returnaddress(i32)

declare i8* @llvm.frameaddress(i32)

define i8* @test1() {
        %X = call i8* @llvm.returnaddress( i32 0 )              ; <i8*> [#uses=1]
        ret i8* %X
}

define i8* @test2() {
        %X = call i8* @llvm.frameaddress( i32 0 )               ; <i8*> [#uses=1]
        ret i8* %X
}

