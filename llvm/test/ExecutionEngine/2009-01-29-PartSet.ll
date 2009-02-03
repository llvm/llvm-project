; RUN: llvm-as %s -o - | lli -force-interpreter | grep FF8F

; ModuleID = 'partset.c.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
@.str = internal constant [4 x i8] c"%X\0A\00"      ; <[4 x i8]*> [#uses=1]

define i32 @main() nounwind  {
entry:
    %part_set = tail call i32 @llvm.part.set.i32.i8( i32 65535, i8 1, i32 7, i32 4 )        ; <i32> [#uses=1]
    %tmp4 = tail call i32 (i8*, ...)* @printf( i8* noalias  getelementptr ([4 x i8]* @.str, i32 0, i64 0), i32 %part_set ) nounwind         ; <i32> [#uses=0]
    ret i32 0
}

declare i32 @llvm.part.set.i32.i8(i32, i8, i32, i32) nounwind readnone

declare i32 @printf(i8*, ...) nounwind
