; Diff file with itself, assert no difference by return code
; RUN: llvm-diff %s %s

; Replace %newvar1 with %newvar2 in the phi node. This can only
; be detected to be different once BB1 has been processed.
; RUN: rm -f %t.ll
; RUN: cat %s | sed -e 's/ %newvar1, %BB1 / %newvar2, %BB1 /' > %t.ll
; RUN: not llvm-diff %s %t.ll 2>&1 | FileCheck --check-prefix DIFFERENT-VAR %s

; DIFFERENT-VAR:      in function func:
; DIFFERENT-VAR-NEXT:   in block %BB0:
; DIFFERENT-VAR-NEXT:     >   %var = phi i32 [ 0, %ENTRY ], [ %newvar2, %BB1 ]
; DIFFERENT-VAR-NEXT:     <   %var = phi i32 [ 0, %ENTRY ], [ %newvar1, %BB1 ]
define i32 @func() {
ENTRY:
  br label %BB0

BB0:
  ; When diffing this phi node, we need to detect whether
  ; %newvar1 is equivalent, which is not known until BB1 has been processed.
  %var = phi i32 [ 0, %ENTRY ], [ %newvar1, %BB1 ]
  %cnd = icmp eq i32 %var, 0
  br i1 %cnd, label %BB1, label %END

BB1:
  %newvar1 = add i32 %var, 1
  %newvar2 = add i32 %var, 2
  br label %BB0

END:
  ; Equivalence of the ret depends on equivalence of %var.
  ; Even if %var differs, we do not report a diff here, because
  ; this is an indirect diff caused by another diff.
  ret i32 %var
}
