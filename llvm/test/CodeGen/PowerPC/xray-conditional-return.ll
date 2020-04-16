; RUN: llc -filetype=asm -o - -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

define void @Foo(i32 signext %a, i32 signext %b) #0 {
; CHECK-LABEL: @Foo
; CHECK:   cmpw
; CHECK-NEXT:   ble 0, [[LABEL:\.[a-zA-Z0-9]+]]
; CHECK-NEXT:   .p2align  3
; CHECK-NEXT: {{\.[a-zA-Z0-9]+}}:
; CHECK-NEXT:   blr
; CHECK-NEXT:   nop
; CHECK-NEXT:   std 0
; CHECK-NEXT:   mflr 0
; CHECK-NEXT:   bl __xray_FunctionExit
; CHECK-NEXT:   nop
; CHECK-NEXT:   mtlr 0
; CHECK-NEXT:   blr
; CHECK-NEXT: [[LABEL]]:
entry:
  %cmp = icmp sgt i32 %a, %b
  br i1 %cmp, label %return, label %if.end

; CHECK:      .p2align  3
; CHECK-NEXT: {{\.[a-zA-Z0-9]+}}:
; CHECK-NEXT:   blr
; CHECK-NEXT:   nop
; CHECK-NEXT:   std 0
; CHECK-NEXT:   mflr 0
; CHECK-NEXT:   bl __xray_FunctionExit
; CHECK-NEXT:   nop
; CHECK-NEXT:   mtlr 0
; CHECK-NEXT:   blr
if.end:
  tail call void @Bar()
  br label %return

return:
  ret void
}

define void @Foo2(i32 signext %a, i32 signext %b) #0 {
; CHECK-LABEL: @Foo2
; CHECK:   cmpw
; CHECK-NEXT:   bge 0, [[LABEL:\.[a-zA-Z0-9]+]]
; CHECK-NEXT:   .p2align  3
; CHECK-NEXT: {{\.[a-zA-Z0-9]+}}:
; CHECK-NEXT:   blr
; CHECK-NEXT:   nop
; CHECK-NEXT:   std 0
; CHECK-NEXT:   mflr 0
; CHECK-NEXT:   bl __xray_FunctionExit
; CHECK-NEXT:   nop
; CHECK-NEXT:   mtlr 0
; CHECK-NEXT:   blr
; CHECK-NEXT: [[LABEL]]:
entry:
  %cmp = icmp slt i32 %a, %b
  br i1 %cmp, label %return, label %if.end

; CHECK:      .p2align  3
; CHECK-NEXT: {{\.[a-zA-Z0-9]+}}:
; CHECK-NEXT:   blr
; CHECK-NEXT:   nop
; CHECK-NEXT:   std 0
; CHECK-NEXT:   mflr 0
; CHECK-NEXT:   bl __xray_FunctionExit
; CHECK-NEXT:   nop
; CHECK-NEXT:   mtlr 0
; CHECK-NEXT:   blr
if.end:
  tail call void @Bar()
  br label %return

return:
  ret void
}

declare void @Bar()

attributes #0 = { "function-instrument"="xray-always" }
