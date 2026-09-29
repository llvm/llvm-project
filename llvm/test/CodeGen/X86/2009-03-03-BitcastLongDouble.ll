; RUN: llc < %s -mtriple=i686--
; PR3686
; rdar://6661799

define i32 @x(i32 %y) nounwind readnone {
entry:
	%tmp14 = zext i32 %y to i80		; <i80> [#uses=1]
	%tmp15 = bitcast i80 %tmp14 to x86_fp80		; <x86_fp80> [#uses=1]
	%add = fadd x86_fp80 %tmp15, 0xK3FFF8000000000000000		; <x86_fp80> [#uses=1]
	%tmp11 = bitcast x86_fp80 %add to i80		; <i80> [#uses=1]
	%tmp10 = trunc i80 %tmp11 to i32		; <i32> [#uses=1]
	ret i32 %tmp10
}

