; RUN: llvm-as < %s | llc -mtriple=thumb-apple-darwin

	%struct.color_sample = type { i32 }
	%struct.ref = type { %struct.color_sample, i16, i16 }

define void @zcvrs() {
	br i1 false, label %bb22, label %UnifiedReturnBlock

bb22:
	br i1 false, label %bb64, label %UnifiedReturnBlock

bb64:
	%tmp67 = urem i32 0, 0
	%tmp69 = icmp slt i32 %tmp67, 10
	%iftmp.13.0 = select i1 %tmp69, i8 48, i8 55
	%tmp75 = add i8 %iftmp.13.0, 0
	store i8 %tmp75, i8* null
	%tmp81 = udiv i32 0, 0
	%tmp83 = icmp eq i32 %tmp81, 0
	br i1 %tmp83, label %bb85, label %bb64

bb85:
	ret void

UnifiedReturnBlock:
	ret void
}
