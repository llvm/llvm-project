; RUN: opt < %s -globalopt -disable-output
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f128:64:128"
target triple = "powerpc-unknown-linux-gnu"
        %struct.empty0 = type {  }
        %struct.es = type { %struct.empty0 }
        %struct.es1 = type { %struct.empty0 }
@aaui1 = internal global [6 x [2 x i32]] [ [2 x i32] [ i32 1, i32 1 ], [2 x i32] [ i32 1, i32 1 ], [2 x i32] [ i32 1, i32 1 ], [2 x i32] [ i32 1, i32 1 ], [2 x i32] [ i32 1, i32 1 ], [2 x i32] [ i32 1, i32 1 ] ]              ; <[6 x [2 x i32]]*> [#uses=1]
@aaui0 = internal global [0 x [2 x i32]] zeroinitializer                ; <[0 x [2 x i32]]*> [#uses=1]

define i8 @func() {
entry:
        %tmp10 = getelementptr [2 x i32], [2 x i32]* getelementptr ([6 x [2 x i32]], [6 x [2 x i32]]* @aaui1, i32 0, i32 0), i32 5, i32 1           ; <i32*> [#uses=1]
        %tmp11 = load i32, i32* %tmp10, align 4              ; <i32> [#uses=1]
        %tmp12 = call i32 (...) @func3( i32* null, i32 0, i32 %tmp11 )         ; <i32> [#uses=0]
        ret i8 undef
}

declare i32 @func3(...)

