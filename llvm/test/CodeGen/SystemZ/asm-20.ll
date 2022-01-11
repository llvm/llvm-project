; Test that asm goto can be compiled.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14

define i32 @c() {
entry:
  callbr void asm sideeffect "j d", "i"(i8* blockaddress(@c, %d))
          to label %asm.fallthrough [label %d]

asm.fallthrough:               ; preds = %entry
  br label %d

d:                             ; preds = %asm.fallthrough, %entry
  ret i32 undef
}
