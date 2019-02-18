; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu -mcpu=g5 | FileCheck %s
; Formerly produced .long, 7320806 (partial)
; CHECK: .byte  22
; CHECK: .byte  21
; CHECK: .byte  20
; CHECK: .byte  3
; CHECK: .byte  25
; CHECK: .byte  24
; CHECK: .byte  23
; CHECK: .byte  3
; CHECK: .byte  28
; CHECK: .byte  27
; CHECK: .byte  26
; CHECK: .byte  3
; CHECK: .byte  31
; CHECK: .byte  30
; CHECK: .byte  29
; CHECK: .byte  3
@baz = common global <16 x i8> zeroinitializer    ; <<16 x i8>*> [#uses=1]

define void @foo(<16 x i8> %x) nounwind ssp {
entry:
  %x_addr = alloca <16 x i8>                      ; <<16 x i8>*> [#uses=2]
  %temp = alloca <16 x i8>                        ; <<16 x i8>*> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  store <16 x i8> %x, <16 x i8>* %x_addr
  store <16 x i8> <i8 22, i8 21, i8 20, i8 3, i8 25, i8 24, i8 23, i8 3, i8 28, i8 27, i8 26, i8 3, i8 31, i8 30, i8 29, i8 3>, <16 x i8>* %temp, align 16
  %0 = load <16 x i8>, <16 x i8>* %x_addr, align 16          ; <<16 x i8>> [#uses=1]
  %1 = load <16 x i8>, <16 x i8>* %temp, align 16            ; <<16 x i8>> [#uses=1]
  %tmp = add <16 x i8> %0, %1                     ; <<16 x i8>> [#uses=1]
  store <16 x i8> %tmp, <16 x i8>* @baz, align 16
  br label %return

return:                                           ; preds = %entry
  ret void
}
