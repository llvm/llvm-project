; RUN: llc < %s -march=ppc32 | not grep cmp

define i32 @seli32_1(i32 %a) {
entry:
	%tmp.1 = icmp slt i32 %a, 0		; <i1> [#uses=1]
	%retval = select i1 %tmp.1, i32 5, i32 0		; <i32> [#uses=1]
	ret i32 %retval
}

define i32 @seli32_2(i32 %a, i32 %b) {
entry:
	%tmp.1 = icmp slt i32 %a, 0		; <i1> [#uses=1]
	%retval = select i1 %tmp.1, i32 %b, i32 0		; <i32> [#uses=1]
	ret i32 %retval
}

define i32 @seli32_3(i32 %a, i16 %b) {
entry:
	%tmp.2 = sext i16 %b to i32		; <i32> [#uses=1]
	%tmp.1 = icmp slt i32 %a, 0		; <i1> [#uses=1]
	%retval = select i1 %tmp.1, i32 %tmp.2, i32 0		; <i32> [#uses=1]
	ret i32 %retval
}

define i32 @seli32_4(i32 %a, i16 %b) {
entry:
	%tmp.2 = zext i16 %b to i32		; <i32> [#uses=1]
	%tmp.1 = icmp slt i32 %a, 0		; <i1> [#uses=1]
	%retval = select i1 %tmp.1, i32 %tmp.2, i32 0		; <i32> [#uses=1]
	ret i32 %retval
}

define i16 @seli16_1(i16 %a) {
entry:
	%tmp.1 = icmp slt i16 %a, 0		; <i1> [#uses=1]
	%retval = select i1 %tmp.1, i16 7, i16 0		; <i16> [#uses=1]
	ret i16 %retval
}

define i16 @seli16_2(i32 %a, i16 %b) {
	%tmp.1 = icmp slt i32 %a, 0		; <i1> [#uses=1]
	%retval = select i1 %tmp.1, i16 %b, i16 0		; <i16> [#uses=1]
	ret i16 %retval
}

define i32 @seli32_a_a(i32 %a) {
	%tmp = icmp slt i32 %a, 1		; <i1> [#uses=1]
	%min = select i1 %tmp, i32 %a, i32 0		; <i32> [#uses=1]
	ret i32 %min
}
