; RUN: llc < %s -mtriple=x86_64--
; PR4583

define i32 @atomic_cmpset_long(i64* %dst, i64 %exp, i64 %src) nounwind ssp noredzone noimplicitfloat {
entry:
	%0 = call i8 asm sideeffect "\09lock ; \09\09\09cmpxchgq $2,$1 ;\09       sete\09$0 ;\09\091:\09\09\09\09# atomic_cmpset_long", "={ax},=*m,r,{ax},*m,~{memory},~{dirflag},~{fpsr},~{flags}"(i64* elementtype(i64) undef, i64 undef, i64 undef, i64* elementtype(i64) undef) nounwind		; <i8> [#uses=0]
	br label %1

; <label>:1		; preds = %entry
	ret i32 undef
}
