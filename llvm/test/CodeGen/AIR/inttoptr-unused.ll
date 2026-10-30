; RUN: llc -mtriple=air -filetype=asm -air-output=air %s -o - | FileCheck %s

; An inttoptr without uses beside one with several: the AIR legalization's
; per-use copies of an address (inttoptr-pun-uses.ll) skip the unused one,
; whose empty use list they once walked past the end of (a crash packaging
; Cycles' intersect_closest).

; CHECK-LABEL: define void @k(
; CHECK-COUNT-2: inttoptr i64 0 to ptr addrspace(1)
define void @k(ptr addrspace(1) %out) {
  %p = inttoptr i64 0 to ptr addrspace(1)
  %dead = inttoptr i64 0 to ptr addrspace(1)
  %a = load i32, ptr addrspace(1) %p, align 4
  store i32 %a, ptr addrspace(1) %p, align 4
  ret void
}
