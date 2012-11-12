; This test case will cause an internal EK_GPRel64BlockAddress to be 
; produced. This was not handled for direct object and an assertion
; to occur. This is a variation on test case test/CodeGen/Mips/do_switch.ll

; RUN: llc < %s -filetype=obj -march=mips -relocation-model=static

; RUN: llc < %s -filetype=obj -march=mips -relocation-model=pic

; RUN: llc < %s -filetype=obj -march=mips64 -relocation-model=pic -mcpu=mips64 -mattr=n64 

define i32 @main() nounwind readnone {
entry:
  %x = alloca i32, align 4                        ; <i32*> [#uses=2]
  store volatile i32 2, i32* %x, align 4
  %0 = load volatile i32* %x, align 4             ; <i32> [#uses=1]

  switch i32 %0, label %bb4 [
    i32 0, label %bb5
    i32 1, label %bb1
    i32 2, label %bb2
    i32 3, label %bb3
  ]

bb1:                                              ; preds = %entry
  ret i32 2

bb2:                                              ; preds = %entry
  ret i32 0

bb3:                                              ; preds = %entry
  ret i32 3

bb4:                                              ; preds = %entry
  ret i32 4

bb5:                                              ; preds = %entry
  ret i32 1
}

