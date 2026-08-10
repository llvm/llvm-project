; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll


; Test "stripped" format where nothing is symbolic... this is how the bytecode
; format looks anyways (except for negative vs positive offsets)...
;
define void @void(i39, i39) {
	add i39 0, 0			; <i39>:3 [#uses=2]
	sub i39 0, 4			; <i39>:4 [#uses=2]
	br label %5

; <label>:5				; preds = %5, %2
	add i39 %0, %1			; <i39>:6 [#uses=2]
	sub i39 %6, %4			; <i39>:7 [#uses=1]
	icmp sle i39 %7, %3		; <i1>:8 [#uses=1]
	br i1 %8, label %9, label %5

; <label>:9				; preds = %5
	add i39 %0, %1			; <i39>:10 [#uses=0]
	sub i39 %6, %4			; <i39>:11 [#uses=1]
	icmp sle i39 %11, %3		; <i1>:12 [#uses=0]
	ret void
}

; This function always returns zero
define i39 @zarro() {
Startup:
	ret i39 0
}
