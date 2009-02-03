; RUN: llvm-as < %s | llc -march=x86-64 | grep call.*memcpy | count 2

declare void @llvm.memcpy.i64(i8*, i8*, i64, i32)

define i8* @my_memcpy(i8* %a, i8* %b, i64 %n) {
entry:
	tail call void @llvm.memcpy.i64( i8* %a, i8* %b, i64 %n, i32 1 )
	ret i8* %a
}

define i8* @my_memcpy2(i64* %a, i64* %b, i64 %n) {
entry:
	%tmp14 = bitcast i64* %a to i8*
	%tmp25 = bitcast i64* %b to i8*
	tail call void @llvm.memcpy.i64(i8* %tmp14, i8* %tmp25, i64 %n, i32 8 )
	ret i8* %tmp14
}
