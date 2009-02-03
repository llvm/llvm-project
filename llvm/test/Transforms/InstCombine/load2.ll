; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep load

@GLOBAL = internal constant [4 x i32] zeroinitializer


define <16 x i8> @foo(<2 x i64> %x) {
entry:
	%tmp = load <16 x i8> * bitcast ([4 x i32]* @GLOBAL to <16 x i8>*)
	ret <16 x i8> %tmp
}

