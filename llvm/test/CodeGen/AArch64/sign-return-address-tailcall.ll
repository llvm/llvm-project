; RUN: llc -mtriple=aarch64 -asm-verbose=0 < %s | FileCheck -DAUTIASP="hint #29" --check-prefixes=COMMON %s
; RUN: llc -mtriple=aarch64 -asm-verbose=0 -aarch64-authenticated-lr-check-method=load                   < %s | FileCheck -DAUTIASP="hint #29" --check-prefixes=COMMON,LDR %s
; RUN: llc -mtriple=aarch64 -asm-verbose=0 -aarch64-authenticated-lr-check-method=high-bits-notbi        < %s | FileCheck -DAUTIASP="hint #29" --check-prefixes=COMMON,BITS-NOTBI,BRK %s
; RUN: llc -mtriple=aarch64 -asm-verbose=0 -aarch64-authenticated-lr-check-method=xpac-hint              < %s | FileCheck -DAUTIASP="hint #29" -DXPACLRI="hint #7" --check-prefixes=COMMON,XPAC,BRK %s
; RUN: llc -mtriple=aarch64 -asm-verbose=0 -aarch64-authenticated-lr-check-method=xpac-hint -mattr=v8.3a < %s | FileCheck -DAUTIASP="autiasp"  -DXPACLRI="xpaclri" --check-prefixes=COMMON,XPAC,BRK %s

define i32 @tailcall_direct() "sign-return-address"="non-leaf" {
; COMMON-LABEL: tailcall_direct:
; COMMON:         str x30, [sp, #-16]!
; COMMON:         ldr x30, [sp], #16
;
; COMMON-NEXT:    [[AUTIASP]]
;
; LDR-NEXT:       ldr w16, [x30]
;
; BITS-NOTBI-NEXT: eor x16, x30, x30, lsl #1
; BITS-NOTBI-NEXT: tbnz x16, #62, .[[FAIL:LBB[_0-9]+]]
;
; XPAC-NEXT:      mov x16, x30
; XPAC-NEXT:      [[XPACLRI]]
; XPAC-NEXT:      cmp x16, x30
; XPAC-NEXT:      b.ne .[[FAIL:LBB[_0-9]+]]
;
; COMMON-NEXT:    b callee
; BRK-NEXT:     .[[FAIL]]:
; BRK-NEXT:       brk #0xc471
  tail call void asm sideeffect "", "~{lr}"()
  %call = tail call i32 @callee()
  ret i32 %call
}

define i32 @tailcall_indirect(ptr %fptr) "sign-return-address"="non-leaf" {
; COMMON-LABEL: tailcall_indirect:
; COMMON:         str x30, [sp, #-16]!
; COMMON:         ldr x30, [sp], #16
;
; COMMON-NEXT:    [[AUTIASP]]
;
; LDR-NEXT:       ldr w16, [x30]
;
; BITS-NOTBI-NEXT: eor x16, x30, x30, lsl #1
; BITS-NOTBI-NEXT: tbnz x16, #62, .[[FAIL:LBB[_0-9]+]]
;
; XPAC-NEXT:      mov x16, x30
; XPAC-NEXT:      [[XPACLRI]]
; XPAC-NEXT:      cmp x16, x30
; XPAC-NEXT:      b.ne .[[FAIL:LBB[_0-9]+]]
;
; COMMON-NEXT:    br x0
; BRK-NEXT:     .[[FAIL]]:
; BRK-NEXT:       brk #0xc471
  tail call void asm sideeffect "", "~{lr}"()
  %call = tail call i32 %fptr()
  ret i32 %call
}

define i32 @tailcall_direct_noframe() "sign-return-address"="non-leaf" {
; COMMON-LABEL: tailcall_direct_noframe:
; COMMON-NEXT:    .cfi_startproc
; COMMON-NEXT:    b callee
  %call = tail call i32 @callee()
  ret i32 %call
}

define i32 @tailcall_indirect_noframe(ptr %fptr) "sign-return-address"="non-leaf" {
; COMMON-LABEL: tailcall_indirect_noframe:
; COMMON-NEXT:    .cfi_startproc
; COMMON-NEXT:    br x0
  %call = tail call i32 %fptr()
  ret i32 %call
}

define i32 @tailcall_direct_noframe_sign_all() "sign-return-address"="all" {
; COMMON-LABEL: tailcall_direct_noframe_sign_all:
; COMMON-NOT:     str{{.*}}x30
; COMMON-NOT:     ldr{{.*}}x30
;
; COMMON:         [[AUTIASP]]
;
; LDR-NEXT:       ldr w16, [x30]
;
; BITS-NOTBI-NEXT: eor x16, x30, x30, lsl #1
; BITS-NOTBI-NEXT: tbnz x16, #62, .[[FAIL:LBB[_0-9]+]]
;
; XPAC-NEXT:      mov x16, x30
; XPAC-NEXT:      [[XPACLRI]]
; XPAC-NEXT:      cmp x16, x30
; XPAC-NEXT:      b.ne .[[FAIL:LBB[_0-9]+]]
;
; COMMON-NEXT:    b callee
; BRK-NEXT:     .[[FAIL]]:
; BRK-NEXT:       brk #0xc471
  %call = tail call i32 @callee()
  ret i32 %call
}

define i32 @tailcall_indirect_noframe_sign_all(ptr %fptr) "sign-return-address"="all" {
; COMMON-LABEL: tailcall_indirect_noframe_sign_all:
; COMMON-NOT:     str{{.*}}x30
; COMMON-NOT:     ldr{{.*}}x30
;
; COMMON:         [[AUTIASP]]
;
; LDR-NEXT:       ldr w16, [x30]
;
; BITS-NOTBI-NEXT: eor x16, x30, x30, lsl #1
; BITS-NOTBI-NEXT: tbnz x16, #62, .[[FAIL:LBB[_0-9]+]]
;
; XPAC-NEXT:      mov x16, x30
; XPAC-NEXT:      [[XPACLRI]]
; XPAC-NEXT:      cmp x16, x30
; XPAC-NEXT:      b.ne .[[FAIL:LBB[_0-9]+]]
;
; COMMON-NEXT:    br x0
; BRK-NEXT:     .[[FAIL]]:
; BRK-NEXT:       brk #0xc471
  %call = tail call i32 %fptr()
  ret i32 %call
}

declare i32 @callee()
