; RUN: opt < %s -passes=instcombine
; PR3468

define x86_fp80 @cast() {
	%tmp = bitcast i80 0 to x86_fp80		; <x86_fp80> [#uses=1]
	ret x86_fp80 %tmp
}

define i80 @invcast() {
	%tmp = bitcast x86_fp80 0xK00000000000000000000 to i80		; <i80> [#uses=1]
	ret i80 %tmp
}
