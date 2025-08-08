; RUN: llc -verify-machineinstrs -mtriple=aarch64-apple-darwin < %s | FileCheck %s --implicit-check-not=.loh --check-prefixes=CHECK,LOH
; RUN: llc -verify-machineinstrs -mtriple=aarch64-apple-darwin -enable-machine-outliner < %s | FileCheck %s --implicit-check-not=.loh --check-prefixes=CHECK,OUTLINE

@A = global i32 0, align 4
@B = global i32 0, align 4

declare void @foo();
declare void @bar(ptr %a);
declare void @goo(ptr %a);

; CHECK-LABEL: _a0:
define void @a0(i32 %a) {

  ; This becomes AdrpAdd when outlining is disabled, otherwise it is outlined
  ; and there should be no LOH.
  %addr = getelementptr inbounds i32, ptr @A, i32 0
  %res = load i32, ptr %addr, align 4
  ; LOH:      [[L0:Lloh.+]]:
  ; LOH-NEXT:   adrp x19, _A@PAGE
  ; LOH-NEXT: [[L1:Lloh.+]]:
  ; LOH-NEXT:   add x19, x19, _A@PAGEOFF

  call void @foo()
  ; OUTLINE:      bl _OUTLINED_FUNCTION_0
  ; OUTLINE-NEXT: mov x0, x19
  ; OUTLINE-NEXT: bl _bar
  call void @bar(ptr %addr)

  ; This becomes AdrpAddStr.
  %addr2 = getelementptr inbounds i32, ptr @B, i32 4
  store i32 %res, ptr %addr2, align 4
  ; CHECK:      [[L2:Lloh.+]]:
  ; CHECK-NEXT:   adrp x8, _B@PAGE
  ; CHECK-NEXT: [[L3:Lloh.+]]:
  ; CHECK-NEXT:   add x8, x8, _B@PAGEOFF
  ; CHECK-NEXT: [[L4:Lloh.+]]:
  ; CHECK-NEXT:   str w20, [x8, #16]
  ret void

  ; LOH-DAG:   .loh AdrpAdd [[L0]], [[L1]]
  ; CHECK-DAG: .loh AdrpAddStr [[L2]], [[L3]], [[L4]]
  ; CHECK:     .cfi_endproc
}

; CHECK-LABEL: _a1:
define i32 @a1(i32 %a) {

  ; This becomes AdrpAdd when outlining is disabled, otherwise it is outlined
  ; and there should be no LOH.
  %addr = getelementptr inbounds i32, ptr @A, i32 0
  %res = load i32, ptr %addr, align 4
  ; LOH:      [[L5:Lloh.+]]:
  ; LOH-NEXT:   adrp x19, _A@PAGE
  ; LOH-NEXT: [[L6:Lloh.+]]:
  ; LOH-NEXT:   add x19, x19, _A@PAGEOFF

  call void @foo()
  ; OUTLINE:      bl _OUTLINED_FUNCTION_0
  ; OUTLINE-NEXT: mov x0, x19
  ; OUTLINE-NEXT: bl _goo
  call void @goo(ptr %addr)
  ret i32 %res

  ; LOH:   .loh AdrpAdd [[L5]], [[L6]]
  ; CHECK: .cfi_endproc
}

; Note: it is not safe to add LOHs to this function as outlined functions do not
; follow calling convention and thus x19 could be live across the call.
; OUTLINE: _OUTLINED_FUNCTION_0:
; OUTLINE:   adrp x19, _A@PAGE
; OUTLINE:   add x19, x19, _A@PAGEOFF
; OUTLINE:   ldr w20, [x19]
; OUTLINE:   b _foo
