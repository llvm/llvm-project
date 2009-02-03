; RUN: llvm-as < %s | llc -march=arm -mattr=+v5t | grep clz

declare i32 @llvm.ctlz.i32(i32)

define i32 @test(i32 %x) {
        %tmp.1 = call i32 @llvm.ctlz.i32( i32 %x )              ; <i32> [#uses=1]
        ret i32 %tmp.1
}
