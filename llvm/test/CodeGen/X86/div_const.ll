; RUN: llvm-as < %s | llc -march=x86 | grep 365384439

define i32 @f9188_mul365384439_shift27(i32 %A) {
        %tmp1 = udiv i32 %A, 1577682821         ; <i32> [#uses=1]
        ret i32 %tmp1
}

