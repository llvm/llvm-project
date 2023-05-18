; RUN: llc < %s -mtriple=arm-linux-gnueabi -mattr=+neon -float-abi=hard

define <16 x i8> @vmulQi8_reg(<16 x i8> %A, <16 x i8> %B) nounwind {
        %tmp1 = mul <16 x i8> %A, %B
        ret <16 x i8> %tmp1
}

define <16 x i8> @f(<16 x i8> %a, <16 x i8> %b) {
        %tmp = call <16 x i8> @g(<16 x i8> %b)
        ret <16 x i8> %tmp
}

declare <16 x i8> @g(<16 x i8>)
