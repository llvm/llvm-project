; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -disable-output 2>/dev/null

define i32 @MTConcat(ptr %a.1) {
	%tmp.961 = getelementptr [3 x i32], ptr %a.1, i64 0, i64 4
	%tmp.97 = load i32, ptr %tmp.961
	%tmp.119 = getelementptr [3 x i32], ptr %a.1, i64 1, i64 0
	%tmp.120 = load i32, ptr %tmp.119
	%tmp.1541 = getelementptr [3 x i32], ptr %a.1, i64 0, i64 4
	%tmp.155 = load i32, ptr %tmp.1541
	ret i32 0
}
