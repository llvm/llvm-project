; RUN: llvm-as < %s | opt -scalarrepl -instcombine | llvm-dis | grep {ret i32 %x}
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"

%pair = type { [1 x i32], i32 }

define i32 @f(i32 %x, i32 %y) {
       %instance = alloca %pair
       %first = getelementptr %pair* %instance, i32 0, i32 0
       %cast = bitcast [1 x i32]* %first to i32*
       store i32 %x, i32* %cast
       %second = getelementptr %pair* %instance, i32 0, i32 1
       store i32 %y, i32* %second
       %v = load i32* %cast
       ret i32 %v
}
