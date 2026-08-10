; Check that differences are reported in the BB processing order
; following the control flow, independent on whether the diff was depending
; on an assumption or not.
;
; Replace %newvar1 with %newvar2 in the phi node. This can only
; be detected to be different once BB2 has been processed, so leads to a assumption
; and is detected to diff later on.
; Also, replace the 1000 by 2000 in BB1, which is detected directly.
;
; RUN: rm -f %t.ll
; RUN: cat %s | sed -e 's/ %newvar1, %BB2 / %newvar2, %BB2 /' | sed -e 's/1000/2000/' > %t.ll
; RUN: not llvm-diff %s %t.ll 2>&1 | FileCheck %s

; CHECK:      in function func:
; CHECK-NEXT:   in block %BB0:
; CHECK-NEXT:     >   %var = phi i32 [ 0, %ENTRY ], [ %newvar2, %BB2 ]
; CHECK-NEXT:     <   %var = phi i32 [ 0, %ENTRY ], [ %newvar1, %BB2 ]
; CHECK-NEXT:   in block %BB1:
; CHECK-NEXT:     >   %diffvar = add i32 %var, 2000
; CHECK-NEXT:     <   %diffvar = add i32 %var, 1000

define i32 @func() {
ENTRY:
  br label %BB0

BB0:
  ; When diffing this phi node, we need to detect whether
  ; %newvar1 is equivalent, which is not known until BB2 has been processed.
  %var = phi i32 [ 0, %ENTRY ], [ %newvar1, %BB2 ]
  %cnd = icmp eq i32 %var, 0
  br i1 %cnd, label %BB1, label %END

BB1:
  %diffvar = add i32 %var, 1000
  br label %BB1

BB2:
  %newvar1 = add i32 %var, 1
  %newvar2 = add i32 %var, 2
  br label %BB0

END:
  ; Equivalence of the ret depends on equivalence of %var.
  ; Even if %var differs, we do not report a diff here, because
  ; this is an indirect diff caused by another diff.
  ret i32 %var
}
