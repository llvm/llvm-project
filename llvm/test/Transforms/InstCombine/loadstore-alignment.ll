; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {, align 16} | count 14

@x = external global <2 x i64>, align 16
@xx = external global [13 x <2 x i64>], align 16

define <2 x i64> @static_hem() {
	%t = getelementptr <2 x i64>* @x, i32 7
	%tmp1 = load <2 x i64>* %t, align 1
	ret <2 x i64> %tmp1
}

define <2 x i64> @hem(i32 %i) {
	%t = getelementptr <2 x i64>* @x, i32 %i
	%tmp1 = load <2 x i64>* %t, align 1
	ret <2 x i64> %tmp1
}

define <2 x i64> @hem_2d(i32 %i, i32 %j) {
	%t = getelementptr [13 x <2 x i64>]* @xx, i32 %i, i32 %j
	%tmp1 = load <2 x i64>* %t, align 1
	ret <2 x i64> %tmp1
}

define <2 x i64> @foo() {
	%tmp1 = load <2 x i64>* @x, align 1
	ret <2 x i64> %tmp1
}

define <2 x i64> @bar() {
	%t = alloca <2 x i64>
        call void @kip(<2 x i64>* %t);
	%tmp1 = load <2 x i64>* %t, align 1
	ret <2 x i64> %tmp1
}

define void @static_hem_store(<2 x i64> %y) {
	%t = getelementptr <2 x i64>* @x, i32 7
	store <2 x i64> %y, <2 x i64>* %t, align 1
        ret void
}

define void @hem_store(i32 %i, <2 x i64> %y) {
	%t = getelementptr <2 x i64>* @x, i32 %i
	store <2 x i64> %y, <2 x i64>* %t, align 1
        ret void
}

define void @hem_2d_store(i32 %i, i32 %j, <2 x i64> %y) {
	%t = getelementptr [13 x <2 x i64>]* @xx, i32 %i, i32 %j
	store <2 x i64> %y, <2 x i64>* %t, align 1
        ret void
}

define void @foo_store(<2 x i64> %y) {
	store <2 x i64> %y, <2 x i64>* @x, align 1
        ret void
}

define void @bar_store(<2 x i64> %y) {
	%t = alloca <2 x i64>
        call void @kip(<2 x i64>* %t);
	store <2 x i64> %y, <2 x i64>* %t, align 1
        ret void
}

declare void @kip(<2 x i64>* %t)
