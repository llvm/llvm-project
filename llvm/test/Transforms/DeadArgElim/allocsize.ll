; RUN: opt < %s -deadargelim -S | FileCheck %s
; PR36867

; CHECK-LABEL: @MagickMallocAligned
; CHECK-NOT: allocsize
define internal i64 @MagickMallocAligned(i64 %DEADARG1, i64 %s) allocsize(1) {
        ret i64 %s
}

define i64 @NeedsArg(i64 %s) {
	%c = call i64 @MagickMallocAligned(i64 0, i64 %s)
	ret i64 %c
}
