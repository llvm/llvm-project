; RUN: llc < %s -march=xcore > %t1.s
; RUN: grep "bl __misaligned_store" %t1.s | count 1
; RUN: grep st16 %t1.s | count 2
; RUN: grep shr %t1.s | count 1

; Byte aligned store. Expands to call to __misaligned_store.
define void @align1(i32* %p, i32 %val) nounwind {
entry:
	store i32 %val, i32* %p, align 1
	ret void
}

; Half word aligned store. Expands to two 16bit stores.
define void @align2(i32* %p, i32 %val) nounwind {
entry:
	store i32 %val, i32* %p, align 2
	ret void
}
