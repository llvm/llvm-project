; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep bitcast
; PR2165

define <1 x i64> @test() {
  %A = bitcast i64 63 to <1 x i64>
  ret <1 x i64> %A
}

