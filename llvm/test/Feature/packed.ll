; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

@foo1 = external global <4 x float>             ; <ptr> [#uses=2]
@foo2 = external global <2 x i32>               ; <ptr> [#uses=2]

define void @main() {
        store <4 x float> < float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00 >, ptr @foo1
        store <2 x i32> < i32 4, i32 4 >, ptr @foo2
        %l1 = load <4 x float>, ptr @foo1           ; <<4 x float>> [#uses=0]
        %l2 = load <2 x i32>, ptr @foo2             ; <<2 x i32>> [#uses=0]
        ret void
}

