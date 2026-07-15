; RUN: llc -mtriple=bpfel -filetype=obj < %s 2>&1 >/dev/null | FileCheck %s

; CHECK: warning: immediate out of range, shall fit in 32 bits
define dso_local void @test_1() naked {
  tail call void asm sideeffect
    "r1 = 40; if r1 == 0x1deadbeef goto +0; r0 = 0; exit;", "~{r0},~{r1}"()
  unreachable
}

; CHECK: warning: immediate out of range, shall fit in 32 bits
define dso_local void @test_2() naked {
  tail call void asm sideeffect
    "r1 = 40; if r1 == 0xffffffff00000000 goto +0; r0 = 0; exit;", "~{r0},~{r1}"()
  unreachable
}
