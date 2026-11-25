; RUN: llc -verify-machineinstrs -mtriple=aarch64-linux-gnu -aarch64-min-jump-table-entries=4 -O2 < %s | FileCheck %s --check-prefixes=CHECK,ISBDSB
; RUN: llc -verify-machineinstrs -mtriple=aarch64-linux-gnu -aarch64-min-jump-table-entries=4 -global-isel -O2 < %s | FileCheck %s --check-prefixes=CHECK,ISBDSB
; RUN: llc -verify-machineinstrs -mtriple=aarch64-linux-gnu -aarch64-min-jump-table-entries=4 -mattr=+sb -O2 < %s | FileCheck %s --check-prefixes=CHECK,SB

; Verify that AArch64 SLS hardening lowers pointer-authenticated indirect calls
; (BLRANoIP) through SLS thunks for all discriminator forms (zero, constant,
; register, blended, and with call arguments) and inserts speculation barriers
; after ptrauth indirectbr (BRA) and hardened jump tables (BR_JumpTable).

declare i64 @llvm.ptrauth.blend(i64, i64)
declare void @f0()
declare void @f1()
declare void @f2()
declare void @f3()
declare void @f4()

@indbr_tbl = constant [2 x ptr] [ptr blockaddress(@indbr_ptrauth, %l1), ptr blockaddress(@indbr_ptrauth, %l2)]

define void @call_ia_zero(ptr %fn) #0 {
; CHECK-LABEL: call_ia_zero:
; CHECK-NOT:   blraa
; CHECK:       bl __llvm_slsblr_thunk_aaz_x0
; CHECK:       ret
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     sb
  call void %fn() [ "ptrauth"(i32 0, i64 0) ]
  ret void
}

define void @call_ib_zero(ptr %fn) #0 {
; CHECK-LABEL: call_ib_zero:
; CHECK-NOT:   blrab
; CHECK:       bl __llvm_slsblr_thunk_abz_x0
; CHECK:       ret
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     sb
  call void %fn() [ "ptrauth"(i32 1, i64 0) ]
  ret void
}

define void @call_ia_const(ptr %fn) #0 {
; CHECK-LABEL: call_ia_const:
; CHECK-NOT:   blraa
; CHECK:       mov w[[DISC:[0-9]+]], #42
; CHECK:       bl __llvm_slsblr_thunk_aa_x0_x[[DISC]]
; CHECK:       ret
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     sb
  call void %fn() [ "ptrauth"(i32 0, i64 42) ]
  ret void
}

define void @call_ia_reg_with_arg(ptr %fn, i64 %arg, i64 %disc) #0 {
; CHECK-LABEL: call_ia_reg_with_arg:
; CHECK-NOT:   blraa
; CHECK:       bl __llvm_slsblr_thunk_aa_x{{[0-9]+}}_x{{[0-9]+}}
; CHECK:       ret
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     sb
  call void %fn(i64 %arg) [ "ptrauth"(i32 0, i64 %disc) ]
  ret void
}

define void @call_ib_blend(ptr %fn, i64 %addr) #0 {
; CHECK-LABEL: call_ib_blend:
; CHECK-NOT:   blrab
; CHECK:       movk x[[DISC:[0-9]+]], #42, lsl #48
; CHECK:       bl __llvm_slsblr_thunk_ab_x{{[0-9]+}}_x[[DISC]]
; CHECK:       ret
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     sb
  %d = call i64 @llvm.ptrauth.blend(i64 %addr, i64 42)
  call void %fn() [ "ptrauth"(i32 1, i64 %d) ]
  ret void
}

define i32 @indbr_ptrauth(i64 %idx) #1 {
; CHECK-LABEL: indbr_ptrauth:
; CHECK:       braa x{{[0-9]+}}, x17
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     sb
entry:
  %gep = getelementptr [2 x ptr], ptr @indbr_tbl, i64 0, i64 %idx
  %p = load ptr, ptr %gep, align 8
  indirectbr ptr %p, [label %l1, label %l2]
l1:
  ret i32 1
l2:
  ret i32 2
}

define void @jt_hardened(i32 %x) #2 {
; CHECK-LABEL: jt_hardened:
; CHECK:       br x16
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     sb
entry:
  switch i32 %x, label %def [
    i32 0, label %b0
    i32 1, label %b1
    i32 2, label %b2
    i32 3, label %b3
    i32 4, label %b4
    i32 5, label %b5
    i32 6, label %b6
  ]
b0:
  call void @f0()
  ret void
b1:
  call void @f1()
  ret void
b2:
  call void @f2()
  ret void
b3:
  call void @f3()
  ret void
b4:
  call void @f4()
  ret void
b5:
  call void @f0()
  call void @f1()
  ret void
b6:
  call void @f2()
  call void @f3()
  ret void
def:
  ret void
}

; CHECK-LABEL: __llvm_slsblr_thunk_aaz_x0:
; CHECK:       mov x16, x0
; CHECK-NEXT:  braaz x16
; CHECK-NEXT:  dsb sy
; CHECK-NEXT:  isb

; CHECK-LABEL: __llvm_slsblr_thunk_abz_x0:
; CHECK:       mov x16, x0
; CHECK-NEXT:  brabz x16
; CHECK-NEXT:  dsb sy
; CHECK-NEXT:  isb

attributes #0 = { nounwind "target-features"="+pauth,+harden-sls-blr,+harden-sls-retbr" }
attributes #1 = { nounwind "ptrauth-indirect-gotos" "target-features"="+pauth,+harden-sls-retbr" }
attributes #2 = { nounwind "aarch64-jump-table-hardening" "target-features"="+harden-sls-retbr" }
