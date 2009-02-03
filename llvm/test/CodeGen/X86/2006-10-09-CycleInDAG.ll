; RUN: llvm-as < %s | llc -march=x86

define void @_ZN13QFSFileEngine4readEPcx() {
	%tmp201 = load i32* null		; <i32> [#uses=1]
	%tmp201.upgrd.1 = sext i32 %tmp201 to i64		; <i64> [#uses=1]
	%tmp202 = load i64* null		; <i64> [#uses=1]
	%tmp203 = add i64 %tmp201.upgrd.1, %tmp202		; <i64> [#uses=1]
	store i64 %tmp203, i64* null
	ret void
}

