; RUN: llvm-as %s -o /dev/null -f

declare i32 @"ArrayRef"([100 x i32] * %Array)

define i32 @"ArrayRef"([100 x i32] * %Array) {
	ret i32 0
}
