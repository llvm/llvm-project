;
; Register constraint "r" shouldn't take long long unless
; The target is 64 bit.
;
;
; RUN: llc -march=mips64el -mcpu=mips64r2 -mattr=n64 < %s | FileCheck %s


define i32 @main() nounwind {
entry:


; r with long long
;CHECK:	#APP
;CHECK:	addi ${{[0-9]+}},${{[0-9]+}},3
;CHECK:	#NO_APP
  tail call i64 asm sideeffect "addi $0,$1,$2", "=r,r,i"(i64 7, i64 3) nounwind
  ret i32 0
}

