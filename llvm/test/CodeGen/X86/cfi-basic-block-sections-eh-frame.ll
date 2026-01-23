; RUN: llc -O0 %s --basic-block-sections=all -mtriple=x86_64 -filetype=obj --frame-pointer=all -o - | llvm-dwarfdump --eh-frame  - | FileCheck --check-prefix=EH_FRAME %s

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z2f3b(i1 zeroext %b) {
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

entry:
  %b.addr = alloca i8, align 1
  %frombool = zext i1 %b to i8
  store i8 %frombool, ptr %b.addr, align 1
  %0 = load i8, ptr %b.addr, align 1
  %tobool = trunc i8 %0 to i1
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @_Z2f1v()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

declare dso_local void @_Z2f1v()
