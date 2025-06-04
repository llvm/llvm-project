; RUN: not llc -mtriple=bpfel -filetype=obj < %s 2>&1 >/dev/null | FileCheck %s

; CHECK: error: immediate out of range, shall fit in 32 bits

define dso_local void @test() naked {
  tail call void asm sideeffect
    "call 7; r1 = r0; r1 <<= 32; call 7; r1 $|= r0; if r1 == 0x1deadbeef goto +1; r0 = 0; exit;",
    "~{r0},~{r1}"()
  unreachable
}
