; RUN: llc < %s -mtriple=s390x-linux-gnu -argext-abi-check
; REQUIRES: asserts
;
; The parameter attributes should match between the call and Function
; declaration as seen by the instruction selector. One of the attributes
; should be present with a normal C-ABI call, but internal/non-C calls can
; also omit the attribute. This seems to work so that the call gets the
; sign/zero extension attribute if the callee prototype has it even if it was
; omitted on the call IR instruction.

define void @caller(i32 %Arg32) {
  ; C Calling Convention
  call void @bar_should_NoExt(i32 %Arg32)
  call void @bar_should_NoExt(i32 noext %Arg32)
  call void @bar_should_NoExt(i32 signext %Arg32)
  call void @bar_should_NoExt(i32 zeroext %Arg32)
  call void @bar_should_SExt(i32 %Arg32)
  call void @bar_should_SExt(i32 noext %Arg32)
  call void @bar_should_SExt(i32 signext %Arg32)
  call void @bar_should_ZExt(i32 %Arg32)
  call void @bar_should_ZExt(i32 noext %Arg32)
  call void @bar_should_ZExt(i32 zeroext %Arg32)

  ; Internal callee but address taken.
  call void @bar_should_NoExt_addrtaken(i32 %Arg32)
  call void @bar_should_NoExt_addrtaken(i32 noext %Arg32)
  call void @bar_should_NoExt_addrtaken(i32 signext %Arg32)
  call void @bar_should_NoExt_addrtaken(i32 zeroext %Arg32)
  call void @bar_should_SExt_addrtaken(i32 %Arg32)
  call void @bar_should_SExt_addrtaken(i32 noext %Arg32)
  call void @bar_should_SExt_addrtaken(i32 signext %Arg32)
  call void @bar_should_ZExt_addrtaken(i32 %Arg32)
  call void @bar_should_ZExt_addrtaken(i32 noext %Arg32)
  call void @bar_should_ZExt_addrtaken(i32 zeroext %Arg32)

  ; Internal callee: No extension is ok.
  call void @bar_internal(i32 %Arg32)
  call void @bar_internal(i32 noext %Arg32)
  call void @bar_internal(i32 signext %Arg32)
  call void @bar_internal(i32 zeroext %Arg32)
  call void @bar_should_NoExt_internal(i32 %Arg32)
  call void @bar_should_NoExt_internal(i32 noext %Arg32)
  call void @bar_should_NoExt_internal(i32 signext %Arg32)
  call void @bar_should_NoExt_internal(i32 zeroext %Arg32)
  call void @bar_should_SExt_internal(i32 %Arg32)
  call void @bar_should_SExt_internal(i32 noext %Arg32)
  call void @bar_should_SExt_internal(i32 signext %Arg32)
  call void @bar_should_ZExt_internal(i32 %Arg32)
  call void @bar_should_ZExt_internal(i32 noext %Arg32)
  call void @bar_should_ZExt_internal(i32 zeroext %Arg32)

  ; Same with a fastcc: No extension is ok.
  call fastcc void @bar_fastcc(i32 %Arg32)
  call fastcc void @bar_fastcc(i32 noext %Arg32)
  call fastcc void @bar_fastcc(i32 signext %Arg32)
  call fastcc void @bar_fastcc(i32 zeroext %Arg32)
  call fastcc void @bar_should_NoExt_fastcc(i32 %Arg32)
  call fastcc void @bar_should_NoExt_fastcc(i32 noext %Arg32)
  call fastcc void @bar_should_NoExt_fastcc(i32 signext %Arg32)
  call fastcc void @bar_should_NoExt_fastcc(i32 zeroext %Arg32)
  call fastcc void @bar_should_SExt_fastcc(i32 %Arg32)
  call fastcc void @bar_should_SExt_fastcc(i32 noext %Arg32)
  call fastcc void @bar_should_SExt_fastcc(i32 signext %Arg32)
  call fastcc void @bar_should_ZExt_fastcc(i32 %Arg32)
  call fastcc void @bar_should_ZExt_fastcc(i32 noext %Arg32)
  call fastcc void @bar_should_ZExt_fastcc(i32 zeroext %Arg32)

  ret void
}

declare void @bar_should_NoExt(i32 noext %Arg32)
declare void @bar_should_SExt(i32 signext %Arg32)
declare void @bar_should_ZExt(i32 zeroext %Arg32)

define internal void @bar_should_NoExt_addrtaken(i32 noext %Arg32) { ret void }
define internal void @bar_should_SExt_addrtaken(i32 signext %Arg32) { ret void }
define internal void @bar_should_ZExt_addrtaken(i32 zeroext %Arg32) { ret void }

define internal void @bar_internal(i32 %Arg32) { ret void }
define internal void @bar_should_NoExt_internal(i32 noext %Arg32) { ret void }
define internal void @bar_should_SExt_internal(i32 signext %Arg32) { ret void }
define internal void @bar_should_ZExt_internal(i32 zeroext %Arg32) { ret void }

declare fastcc void @bar_fastcc(i32 %Arg32)
declare fastcc void @bar_should_NoExt_fastcc(i32 noext %Arg32)
declare fastcc void @bar_should_SExt_fastcc(i32 signext %Arg32)
declare fastcc void @bar_should_ZExt_fastcc(i32 zeroext %Arg32)

declare void @ExtFun(ptr %FunPtr)
define void @foo() {
  call void @ExtFun(ptr @bar_should_NoExt_addrtaken)
  call void @ExtFun(ptr @bar_should_SExt_addrtaken)
  call void @ExtFun(ptr @bar_should_ZExt_addrtaken)
  ret void
}
