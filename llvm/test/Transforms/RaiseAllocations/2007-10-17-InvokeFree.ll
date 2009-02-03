; RUN: llvm-as <%s | opt -raiseallocs -stats -disable-output |&  \
; RUN:  not grep {Number of allocations raised}
define void @foo() {
entry:
	%buffer = alloca i16*
	%tmp = load i16** %buffer, align 8
	invoke i32(...)* @free(i16* %tmp)
		to label %invcont unwind label %unwind
invcont:
	br label %finally
unwind:
	br label %finally
finally:
	ret void
}
declare i32 @free(...)

