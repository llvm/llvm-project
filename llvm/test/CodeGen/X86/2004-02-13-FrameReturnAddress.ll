; RUN: llvm-as < %s | llc -march=x86 | grep {(%esp}

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

