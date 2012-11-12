; RUN: llc < %s -march=thumb

; Test Thumb-mode "I" constraint, for ADD immediate.
define i32 @testI(i32 %x) {
	%y = call i32 asm "add $0, $1, $2", "=r,r,I"( i32 %x, i32 255 ) nounwind
	ret i32 %y
}

; Test Thumb-mode "J" constraint, for negated ADD immediates.
define void @testJ() {
	tail call void asm sideeffect ".word $0", "J"( i32 -255 ) nounwind
	ret void
}

; Test Thumb-mode "K" constraint, for compatibility with GCC's internal use.
define void @testK() {
	tail call void asm sideeffect ".word $0", "K"( i32 65280 ) nounwind
	ret void
}

; Test Thumb-mode "L" constraint, for 3-operand ADD immediates.
define i32 @testL(i32 %x) {
	%y = call i32 asm "add $0, $1, $2", "=r,r,L"( i32 %x, i32 -7 ) nounwind
	ret i32 %y
}

; Test Thumb-mode "M" constraint, for "ADD r = sp + imm".
define i32 @testM() {
	%y = call i32 asm "add $0, sp, $1", "=r,M"( i32 1020 ) nounwind
	ret i32 %y
}

; Test Thumb-mode "N" constraint, for values between 0 and 31.
define i32 @testN(i32 %x) {
	%y = call i32 asm "lsl $0, $1, $2", "=r,r,N"( i32 %x, i32 31 ) nounwind
	ret i32 %y
}

; Test Thumb-mode "O" constraint, for "ADD sp = sp + imm".
define void @testO() {
	tail call void asm sideeffect "add sp, sp, $0; add sp, sp, $1", "O,O"( i32 -508, i32 508 ) nounwind
        ret void
}
