; RUN: llc < %s -mtriple=arm-apple-darwin -relocation-model=dynamic-no-pic -mcpu=cortex-a8 -asm-verbose=false | FileCheck %s

declare void @bar(i32)
declare void @car(i32)
declare void @dar(i32)
declare void @ear(i32)
declare void @far(i32)
declare i1 @qux()

@GHJK = global i32 0

declare i8* @choose(i8*, i8*)

; BranchFolding should tail-duplicate the indirect jump to avoid
; redundant branching.

; CHECK: tail_duplicate_me:
; CHECK:      qux
; CHECK:      movw r{{[0-9]+}}, :lower16:_GHJK
; CHECK:      movt r{{[0-9]+}}, :upper16:_GHJK
; CHECK:      str r
; CHECK-NEXT: bx r
; CHECK:      qux
; CHECK:      movw r{{[0-9]+}}, :lower16:_GHJK
; CHECK:      movt r{{[0-9]+}}, :upper16:_GHJK
; CHECK:      str r
; CHECK-NEXT: bx r
; CHECK:      movw r{{[0-9]+}}, :lower16:_GHJK
; CHECK:      movt r{{[0-9]+}}, :upper16:_GHJK
; CHECK:      str r
; CHECK-NEXT: bx r

define void @tail_duplicate_me() nounwind {
entry:
  %a = call i1 @qux()
  %c = call i8* @choose(i8* blockaddress(@tail_duplicate_me, %return),
                        i8* blockaddress(@tail_duplicate_me, %altret))
  br i1 %a, label %A, label %next
next:
  %b = call i1 @qux()
  br i1 %b, label %B, label %C

A:
  call void @bar(i32 0)
  store i32 0, i32* @GHJK
  br label %M

B:
  call void @car(i32 1)
  store i32 0, i32* @GHJK
  br label %M

C:
  call void @dar(i32 2)
  store i32 0, i32* @GHJK
  br label %M

M:
  indirectbr i8* %c, [label %return, label %altret]

return:
  call void @ear(i32 1000)
  ret void
altret:
  call void @far(i32 1001)
  ret void
}
