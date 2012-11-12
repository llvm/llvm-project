; RUN: llc < %s -march=arm

; Test ARM-mode "I" constraint, for any Data Processing immediate.
define i32 @testI(i32 %x) {
	%y = call i32 asm "add $0, $1, $2", "=r,r,I"( i32 %x, i32 65280 ) nounwind
	ret i32 %y
}

; Test ARM-mode "J" constraint, for compatibility with unknown use in GCC.
define void @testJ() {
	tail call void asm sideeffect ".word $0", "J"( i32 4080 ) nounwind
	ret void
}

; Test ARM-mode "K" constraint, for bitwise inverted Data Processing immediates.
define void @testK() {
	tail call void asm sideeffect ".word $0", "K"( i32 16777215 ) nounwind
	ret void
}

; Test ARM-mode "L" constraint, for negated Data Processing immediates.
define void @testL() {
	tail call void asm sideeffect ".word $0", "L"( i32 -65280 ) nounwind
	ret void
}

; Test ARM-mode "M" constraint, for value between 0 and 32.
define i32 @testM(i32 %x) {
	%y = call i32 asm "lsl $0, $1, $2", "=r,r,M"( i32 %x, i32 31 ) nounwind
	ret i32 %y
}
