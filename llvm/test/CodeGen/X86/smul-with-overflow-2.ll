; RUN: llvm-as < %s | llc -march=x86 | grep mul | count 1
; RUN: llvm-as < %s | llc -march=x86 | grep add | count 3

define i32 @t1(i32 %a, i32 %b) nounwind readnone {
entry:
        %tmp0 = add i32 %b, %a
	%tmp1 = call { i32, i1 } @llvm.smul.with.overflow.i32(i32 %tmp0, i32 2)
	%tmp2 = extractvalue { i32, i1 } %tmp1, 0
	ret i32 %tmp2
}

define i32 @t2(i32 %a, i32 %b) nounwind readnone {
entry:
        %tmp0 = add i32 %b, %a
	%tmp1 = call { i32, i1 } @llvm.smul.with.overflow.i32(i32 %tmp0, i32 4)
	%tmp2 = extractvalue { i32, i1 } %tmp1, 0
	ret i32 %tmp2
}

declare { i32, i1 } @llvm.smul.with.overflow.i32(i32, i32) nounwind
