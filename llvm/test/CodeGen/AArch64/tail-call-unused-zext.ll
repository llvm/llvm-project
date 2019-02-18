; RUN: llc -mtriple=arm64--- -stop-after=expand-isel-pseudos -o - %s | FileCheck %s

; Check that we ignore the zeroext attribute on the return type of the tail
; call, since the return value is unused. This happens during CodeGenPrepare in
; dupRetToEnableTailCallOpts, which calls attributesPermitTailCall to check if
; the attributes of the caller and the callee match.

declare zeroext i1 @zcallee()
define void @zcaller() {
; CHECK-LABEL: name: zcaller
entry:
  br i1 undef, label %calllabel, label %retlabel
calllabel:
; CHECK: bb.1.calllabel:
; CHECK-NOT: BL @zcallee
; CHECK-NEXT: TCRETURNdi @zcallee
  %unused_result = tail call zeroext i1 @zcallee()
  br label %retlabel
retlabel:
  ret void
}

declare signext i1 @scallee()
define void @scaller() {
; CHECK-LABEL: name: scaller
entry:
  br i1 undef, label %calllabel, label %retlabel
calllabel:
; CHECK: bb.1.calllabel:
; CHECK-NOT: BL @scallee
; CHECK-NEXT: TCRETURNdi @scallee
  %unused_result = tail call signext i1 @scallee()
  br label %retlabel
retlabel:
  ret void
}
