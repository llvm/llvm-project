; RUN: llvm-as < %s | llc -march=x86 -mcpu=yonah | grep add | grep 16

	%struct.S = type { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>,
                           <2 x i64> }

define i32 @main() nounwind  {
entry:
	%s = alloca %struct.S		; <%struct.S*> [#uses=2]
	%tmp15 = getelementptr %struct.S* %s, i32 0, i32 0		; <<2 x i64>*> [#uses=1]
	store <2 x i64> < i64 8589934595, i64 1 >, <2 x i64>* %tmp15, align 16
	call void @t( i32 1, %struct.S* byval  %s ) nounwind 
	ret i32 0
}

declare void @t(i32, %struct.S* byval )
