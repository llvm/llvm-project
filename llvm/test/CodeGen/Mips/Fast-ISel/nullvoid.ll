; RUN: llc -march=mipsel -relocation-model=pic -O0 -fast-isel-abort=1 -mcpu=mips32r2 \
; RUN:     < %s | FileCheck %s
; RUN: llc -march=mipsel -relocation-model=pic -O0 -fast-isel-abort=1 -mcpu=mips32 \
; RUN:     < %s | FileCheck %s

; Function Attrs: nounwind
define void @foo() {
entry:
  ret void
; CHECK: jr	$ra
}
