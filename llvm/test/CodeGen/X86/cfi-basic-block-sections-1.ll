; RUN: llc -O0 %s --basic-block-sections=all -mtriple=x86_64 -filetype=asm --frame-pointer=all -o - | FileCheck --check-prefix=SECTIONS_CFI %s
; RUN: llc -O0 %s --basic-block-sections=all -mtriple=x86_64 -filetype=asm --frame-pointer=none -o - | FileCheck --check-prefix=SECTIONS_NOFP_CFI %s
; RUN: llc -O0 %s --basic-block-sections=all -mtriple=x86_64 -filetype=obj --frame-pointer=all -o - | llvm-dwarfdump --eh-frame  - | FileCheck --check-prefix=EH_FRAME %s

;; void f1();
;; void f3(bool b) {
;;   if (b)
;;     f1();
;; }


; SECTIONS_CFI: _Z2f3b:
; SECTIONS_CFI: .cfi_startproc
; SECTIONS_CFI: .cfi_def_cfa_offset 16
; SECTIONS_CFI: .cfi_offset %rbp, -16
; SECTIONS_CFI: .cfi_def_cfa_register %rbp
; SECTIONS_CFI: .cfi_endproc

; SECTIONS_CFI: _Z2f3b.__part.1:
; SECTIONS_CFI-NEXT: .cfi_startproc
; SECTIONS_CFI-NEXT: .cfi_def_cfa %rbp, 16
; SECTIONS_CFI-NEXT: .cfi_offset %rbp, -16
; SECTIONS_CFI: .cfi_endproc

; SECTIONS_CFI: _Z2f3b.__part.2:
; SECTIONS_CFI-NEXT: .cfi_startproc
; SECTIONS_CFI-NEXT: .cfi_def_cfa %rbp, 16
; SECTIONS_CFI-NEXT: .cfi_offset %rbp, -16
; SECTIONS_CFI: .cfi_def_cfa
; SECTIONS_CFI: .cfi_endproc


; SECTIONS_NOFP_CFI: _Z2f3b:
; SECTIONS_NOFP_CFI: .cfi_startproc
; SECTIONS_NOFP_CFI: .cfi_def_cfa_offset 16
; SECTIONS_NOFP_CFI: .cfi_endproc

; SECTIONS_NOFP_CFI: _Z2f3b.__part.1:
; SECTIONS_NOFP_CFI-NEXT: .cfi_startproc
; SECTIONS_NOFP_CFI-NEXT: .cfi_def_cfa %rsp, 16
; SECTIONS_NOFP_CFI: .cfi_endproc

; SECTIONS_NOFP_CFI: _Z2f3b.__part.2:
; SECTIONS_NOFP_CFI-NEXT: .cfi_startproc
; SECTIONS_NOFP_CFI-NEXT: .cfi_def_cfa %rsp, 16
; SECTIONS_NOFP_CFI: .cfi_endproc


;; There must be 1 CIE and 3 FDEs.

; EH_FRAME: CIE
; EH_FRAME: DW_CFA_def_cfa
; EH_FRAME: DW_CFA_offset

; EH_FRAME: FDE cie=
; EH_FRAME: DW_CFA_def_cfa_offset
; EH_FRAME: DW_CFA_offset
; EH_FRAME: DW_CFA_def_cfa_register

; EH_FRAME: FDE cie=
; EH_FRAME: DW_CFA_def_cfa
; EH_FRAME: DW_CFA_offset

; EH_FRAME: FDE cie=
; EH_FRAME: DW_CFA_def_cfa
; EH_FRAME: DW_CFA_offset

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z2f3b(i1 zeroext %b) {
entry:
  %b.addr = alloca i8, align 1
  %frombool = zext i1 %b to i8
  store i8 %frombool, i8* %b.addr, align 1
  %0 = load i8, i8* %b.addr, align 1
  %tobool = trunc i8 %0 to i1
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @_Z2f1v()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

declare dso_local void @_Z2f1v()
