; RUN: llc -mtriple=aarch64 -asm-verbose=0 < %s | FileCheck -DAUTIASP="hint #29" --check-prefixes=COMMON %s
; RUN: llc -mtriple=aarch64 -asm-verbose=0 -aarch64-authenticated-lr-check-method=load                   < %s | FileCheck -DAUTIASP="hint #29" --check-prefixes=COMMON,LDR %s
; RUN: llc -mtriple=aarch64 -asm-verbose=0 -aarch64-authenticated-lr-check-method=high-bits-notbi        < %s | FileCheck -DAUTIASP="hint #29" --check-prefixes=COMMON,BITS-NOTBI,BRK %s
; RUN: llc -mtriple=aarch64 -asm-verbose=0 -aarch64-authenticated-lr-check-method=xpac-hint              < %s | FileCheck -DAUTIASP="hint #29" -DXPACLRI="hint #7" --check-prefixes=COMMON,XPAC,BRK %s
; RUN: llc -mtriple=aarch64 -asm-verbose=0 -aarch64-authenticated-lr-check-method=xpac-hint -mattr=v8.3a < %s | FileCheck -DAUTIASP="autiasp"  -DXPACLRI="xpaclri" --check-prefixes=COMMON,XPAC,BRK %s
; RUN: llc -mtriple=aarch64 -asm-verbose=0 -aarch64-authenticated-lr-check-method=xpac      -mattr=v8.3a < %s | FileCheck -DAUTIASP="autiasp"  --check-prefixes=COMMON,XPAC83,BRK %s

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
; BITS-NOTBI-NEXT: tbz x16, #62, .[[GOOD:Lauth_success[_0-9]+]]
;
; XPAC-NEXT:      mov x16, x30
; XPAC-NEXT:      [[XPACLRI]]
; XPAC-NEXT:      cmp x30, x16
; XPAC-NEXT:      b.eq .[[GOOD:Lauth_success[_0-9]+]]
;
; XPAC83-NEXT:    mov x16, x30
; XPAC83-NEXT:    xpaci x16
; XPAC83-NEXT:    cmp x30, x16
; XPAC83-NEXT:    b.eq .[[GOOD:Lauth_success[_0-9]+]]
;
; BRK-NEXT:       brk #0xc470
; BRK-NEXT:     .[[GOOD]]:
; COMMON-NEXT:    b callee
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
; BITS-NOTBI-NEXT: tbz x16, #62, .[[GOOD:Lauth_success[_0-9]+]]
;
; XPAC-NEXT:      mov x16, x30
; XPAC-NEXT:      [[XPACLRI]]
; XPAC-NEXT:      cmp x30, x16
; XPAC-NEXT:      b.eq .[[GOOD:Lauth_success[_0-9]+]]
;
; XPAC83-NEXT:    mov x16, x30
; XPAC83-NEXT:    xpaci x16
; XPAC83-NEXT:    cmp x30, x16
; XPAC83-NEXT:    b.eq .[[GOOD:Lauth_success[_0-9]+]]
;
; BRK-NEXT:       brk #0xc470
; BRK-NEXT:     .[[GOOD]]:
; COMMON-NEXT:    br x0
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
; BITS-NOTBI-NEXT: tbz x16, #62, .[[GOOD:Lauth_success[_0-9]+]]
;
; XPAC-NEXT:      mov x16, x30
; XPAC-NEXT:      [[XPACLRI]]
; XPAC-NEXT:      cmp x30, x16
; XPAC-NEXT:      b.eq .[[GOOD:Lauth_success[_0-9]+]]
;
; XPAC83-NEXT:    mov x16, x30
; XPAC83-NEXT:    xpaci x16
; XPAC83-NEXT:    cmp x30, x16
; XPAC83-NEXT:    b.eq .[[GOOD:Lauth_success[_0-9]+]]
;
; BRK-NEXT:       brk #0xc470
; BRK-NEXT:     .[[GOOD]]:
; COMMON-NEXT:    b callee
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
; BITS-NOTBI-NEXT: tbz x16, #62, .[[GOOD:Lauth_success[_0-9]+]]
;
; XPAC-NEXT:      mov x16, x30
; XPAC-NEXT:      [[XPACLRI]]
; XPAC-NEXT:      cmp x30, x16
; XPAC-NEXT:      b.eq .[[GOOD:Lauth_success[_0-9]+]]
;
; XPAC83-NEXT:    mov x16, x30
; XPAC83-NEXT:    xpaci x16
; XPAC83-NEXT:    cmp x30, x16
; XPAC83-NEXT:    b.eq .[[GOOD:Lauth_success[_0-9]+]]
;
; BRK-NEXT:       brk #0xc470
; BRK-NEXT:     .[[GOOD]]:
; COMMON-NEXT:    br x0
  %call = tail call i32 %fptr()
  ret i32 %call
}

define i32 @tailcall_ib_key() "sign-return-address"="all" "sign-return-address-key"="b_key" {
; COMMON-LABEL: tailcall_ib_key:
;
; BRK:            brk #0xc471
; BRK-NEXT:     .{{Lauth_success.*}}:
; COMMON:         b callee
  tail call void asm sideeffect "", "~{lr}"()
  %call = tail call i32 @callee()
  ret i32 %call
}

define i32 @tailcall_two_branches(i1 %0) "sign-return-address"="all" {
; COMMON-LABEL:    tailcall_two_branches:
; COMMON:            tbz w0, #0, .[[ELSE:LBB[_0-9]+]]
; COMMON:            str x30, [sp, #-16]!
; COMMON:            bl callee2
; COMMON:            ldr x30, [sp], #16
; COMMON-NEXT:       [[AUTIASP]]
; COMMON-NEXT:     .[[ELSE]]:

; LDR-NEXT:          ldr w16, [x30]
;
; BITS-NOTBI-NEXT:   eor x16, x30, x30, lsl #1
; BITS-NOTBI-NEXT:   tbz x16, #62, .[[GOOD:Lauth_success[_0-9]+]]
;
; XPAC-NEXT:         mov x16, x30
; XPAC-NEXT:         [[XPACLRI]]
; XPAC-NEXT:         cmp x30, x16
; XPAC-NEXT:         b.eq .[[GOOD:Lauth_success[_0-9]+]]
;
; XPAC83-NEXT:       mov x16, x30
; XPAC83-NEXT:       xpaci x16
; XPAC83-NEXT:       cmp x30, x16
; XPAC83-NEXT:       b.eq .[[GOOD:Lauth_success[_0-9]+]]
;
; BRK-NEXT:          brk #0xc470
; BRK-NEXT:        .[[GOOD]]:
; COMMON-NEXT:       b callee
  br i1 %0, label %2, label %3
2:
  call void @callee2()
  br label %3
3:
  %call = tail call i32 @callee()
  ret i32 %call
}

declare i32 @callee()
declare void @callee2()
