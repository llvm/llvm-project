; RUN: llc < %s
target datalayout = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8"
target triple = "msp430-generic-generic"

define void @imm() nounwind {
        call void asm sideeffect "bic\09$0,r2", "i"(i16 32) nounwind
        ret void
}

define void @reg(i16 %a) nounwind {
        call void asm sideeffect "bic\09$0,r2", "r"(i16 %a) nounwind
        ret void
}

@foo = global i16 0, align 2

define void @immmem() nounwind {
        call void asm sideeffect "bic\09$0,r2", "i"(i16* getelementptr(i16* @foo, i32 1)) nounwind
        ret void
}

define void @mem() nounwind {
        %fooval = load i16* @foo
        call void asm sideeffect "bic\09$0,r2", "m"(i16 %fooval) nounwind
        ret void
}
