; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s
;; Emitting an object checks each sled's size against getInstSizeInBytes
;; (assertions builds only).
; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu -filetype=obj < %s -o /dev/null

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

;; If-conversion predicates the return on a single CR bit (bclr 12 / bclr 4).
define void @Bit(i1 %a, i1 %b) #0 {
; CHECK-LABEL: @Bit
; CHECK:        andi. 3, 3, 1
; CHECK-NEXT:   bc 4, 1, [[LABEL:\.[a-zA-Z0-9]+]]
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
  %c = and i1 %a, %b
  br i1 %c, label %return, label %if.end

if.end:
  tail call void @Bar()
  br label %return

return:
  ret void
}

define void @BitN(i1 %a, i1 %b) #0 {
; CHECK-LABEL: @BitN
; CHECK:        andi. 3, 3, 1
; CHECK-NEXT:   bc 12, 1, [[LABEL:\.[a-zA-Z0-9]+]]
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
  %c = and i1 %a, %b
  br i1 %c, label %if.end, label %return

if.end:
  tail call void @Bar()
  br label %return

return:
  ret void
}

;; The loop exit returns on CTR (bdzlr).
define i32 @Ctr(i32 %n, ptr %p, i32 %limit) #0 {
; CHECK-LABEL: @Ctr
; CHECK:        bdnz [[LABEL:\.[a-zA-Z0-9]+]]
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
  %cmp.entry = icmp sgt i32 %n, 0
  br i1 %cmp.entry, label %preheader, label %return

preheader:
  %n64 = sext i32 %n to i64
  br label %body

latch:
  %cmp = icmp slt i64 %iv.next, %n64
  br i1 %cmp, label %body, label %return

body:
  %iv = phi i64 [ 0, %preheader ], [ %iv.next, %latch ]
  %addr = getelementptr inbounds i32, ptr %p, i64 %iv
  %v = load i32, ptr %addr
  %lt = icmp slt i32 %v, %limit
  %iv.next = add nuw nsw i64 %iv, 1
  br i1 %lt, label %latch, label %return

return:
  %r = phi i32 [ 0, %entry ], [ 0, %latch ], [ 1, %body ]
  ret i32 %r
}

declare void @Bar()

attributes #0 = { "function-instrument"="xray-always" }
