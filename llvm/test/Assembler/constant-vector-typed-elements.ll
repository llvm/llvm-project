; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; make sure typed constant vectors still parse fine after the apsint guard

@my_global = external global i32

; CHECK: @g1 = constant <2 x i16> <i16 123, i16 456>
@g1 = constant <2 x i16> <i16 123, i16 456>

; CHECK: @g2 = constant <2 x float> <float 1.000000e+00, float 2.000000e+00>
@g2 = constant <2 x float> <float 1.0, float 2.0>

; CHECK: @g3 = constant <{ i32, i32 }> <{ i32 1, i32 2 }>
@g3 = constant <{ i32, i32 }> <{ i32 1, i32 2 }>

; CHECK: @g4 = constant <2 x ptr> <ptr @my_global, ptr @my_global>
@g4 = constant <2 x ptr> <ptr @my_global, ptr @my_global>

; CHECK: @g7 = constant <4 x i32> splat (i32 7)
@g7 = constant <4 x i32> splat (i32 7)
