; RUN: llc -mtriple=thumbv5-none-linux-gnueabi -no-integrated-as %s -o /dev/null

; Test thumb-mode "I" constraint, for any Data Processing immediate.
define void @testI() {
	tail call void asm sideeffect ".word $0", "I"( i32 255 ) nounwind
	ret void
}

; Test thumb-mode "J" constraint, for compatibility with unknown use in GCC.
define void @testJ() {
	tail call void asm sideeffect ".word $0", "J"( i32 -254 ) nounwind
	ret void
}

; Test thumb-mode "L" constraint, for negated Data Processing immediates.
define void @testL() {
	tail call void asm sideeffect ".word $0", "L"( i32 -7 ) nounwind
	ret void
}

