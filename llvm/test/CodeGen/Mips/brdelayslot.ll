; RUN: llc -march=mipsel -O0 < %s | FileCheck %s -check-prefix=None
; RUN: llc -march=mipsel < %s | FileCheck %s -check-prefix=Default

define void @foo1() nounwind {
entry:
; Default:     jalr 
; Default-NOT: nop 
; Default:     jr 
; Default-NOT: nop
; Default:     .end
; None: jalr 
; None: nop 
; None: jr 
; None: nop
; None: .end

  tail call void @foo2(i32 3) nounwind
  ret void
}

declare void @foo2(i32)

; Check that cvt.d.w goes into jalr's delay slot.
;
define void @foo3(i32 %a) nounwind {
entry:
; Default:     foo3:
; Default:     jalr
; Default:     cvt.d.w

  %conv = sitofp i32 %a to double
  tail call void @foo4(double %conv) nounwind
  ret void
}

declare void @foo4(double)

