; RUN: llc -fast-isel-sink-local-values -march=mipsel -relocation-model=pic -O0 -fast-isel-abort=3 -mcpu=mips32r2 \
; RUN:     < %s | FileCheck %s
; RUN: llc -fast-isel-sink-local-values -march=mipsel -relocation-model=pic -O0 -fast-isel-abort=3 -mcpu=mips32 \
; RUN:     < %s | FileCheck %s

@abcd = external global i32

; Function Attrs: nounwind
define void @foo()  {
entry:
  store i32 12345, i32* @abcd, align 4
; CHECK: 	lw	$[[REG2:[0-9]+]], %got(abcd)(${{[0-9]+}})
; CHECK: 	addiu	$[[REG1:[0-9]+]], $zero, 12345
; CHECK: 	sw	$[[REG1]], 0($[[REG2]])
  ret void
}

