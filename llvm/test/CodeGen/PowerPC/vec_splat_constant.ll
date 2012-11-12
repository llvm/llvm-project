; RUN: llc < %s -march=ppc32 -mtriple=powerpc-apple-darwin -mcpu=g5 | FileCheck %s
; Formerly incorrectly inserted vsldoi (endian confusion)

@baz = common global <16 x i8> zeroinitializer    ; <<16 x i8>*> [#uses=1]

define void @foo(<16 x i8> %x) nounwind ssp {
entry:
; CHECK: _foo:
; CHECK-NOT: vsldoi
  %x_addr = alloca <16 x i8>                      ; <<16 x i8>*> [#uses=2]
  %temp = alloca <16 x i8>                        ; <<16 x i8>*> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  store <16 x i8> %x, <16 x i8>* %x_addr
  store <16 x i8> <i8 0, i8 0, i8 0, i8 14, i8 0, i8 0, i8 0, i8 14, i8 0, i8 0, i8 0, i8 14, i8 0, i8 0, i8 0, i8 14>, <16 x i8>* %temp, align 16
  %0 = load <16 x i8>* %x_addr, align 16          ; <<16 x i8>> [#uses=1]
  %1 = load <16 x i8>* %temp, align 16            ; <<16 x i8>> [#uses=1]
  %tmp = add <16 x i8> %0, %1                     ; <<16 x i8>> [#uses=1]
  store <16 x i8> %tmp, <16 x i8>* @baz, align 16
  br label %return

return:                                           ; preds = %entry
  ret void
; CHECK: blr
}
