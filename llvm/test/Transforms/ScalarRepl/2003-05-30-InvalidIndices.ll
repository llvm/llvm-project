; RUN: llvm-as < %s | opt -scalarrepl

define void @main() {
	%E = alloca { { i32, float, double, i64 }, { i32, float, double, i64 } }	; <{ { i32, float, double, i64 }, { i32, float, double, i64 } }*> [#uses=1]
	%tmp.151 = getelementptr { { i32, float, double, i64 }, { i32, float, double, i64 } }* %E, i64 0, i32 1, i32 3		; <i64*> [#uses=0]
	ret void
}

