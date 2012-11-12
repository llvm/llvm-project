; RUN: llc -march=mipsel -relocation-model=pic < %s | FileCheck %s -check-prefix=CHECK-PIC
; RUN: llc -march=mipsel -relocation-model=static < %s | FileCheck %s -check-prefix=CHECK-STATIC

define void @count(i32 %x, i32 %y, i32 %z) noreturn nounwind readnone {
entry:
  br label %bosco

bosco:                                            ; preds = %bosco, %entry
  br label %bosco
}

; CHECK-PIC: b	$BB0_1
; CHECK-STATIC: j	$BB0_1
