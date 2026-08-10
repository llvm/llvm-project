; RUN: llc < %s -mtriple=aarch64-linux-gnu | FileCheck %s

; Test function which compares two integers and returns the value of
; the overflow flag, by using an asm goto to make the asm block branch
; based on that flag, and then a phi to set the return value based on
; whether the branch was taken.
define i32 @overflow(i64 %a, i64 %b) #0 {
asm:
  callbr void asm sideeffect "cmp $0, $1 \0A\09 b.vs ${2:l}",
          "r,r,!i,~{cc}"(i64 %a, i64 %b)
          to label %fallthrough [label %indirect]

indirect:
  br label %fallthrough

fallthrough:
  ; Return 1 if we came via the 'indirect' block (because the b.vs was
  ; taken), and 0 if we came straight from the asm block (because it
  ; was untaken).
  %retval = phi i32 [0, %asm], [1, %indirect]
  ret i32 %retval
}

; CHECK: overflow:
; CHECK-NEXT: .cfi_startproc
; CHECK-NEXT: // %bb.{{[0-9]+}}:
; CHECK-NEXT: bti c
; CHECK-NEXT: //APP
; CHECK-NEXT: cmp x0, x1
; CHECK-NEXT: b.vs [[LABEL:\.[A-Za-z0-9_]+]]
; CHECK-NEXT: //NO_APP
; CHECK-NEXT: // %bb.{{[0-9]+}}:
; CHECK-NEXT: mov w0, wzr
; CHECK-NEXT: ret
; CHECK-NEXT: [[LABEL]]:
; CHECK-NOT:  bti
; CHECK:      mov w0, #1
; CHECK-NEXT: ret

attributes #0 = { "branch-target-enforcement" "target-features"="+bti" }
