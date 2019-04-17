; Test to make sure that load from the same address as a store and appears after the store prevents the store from being sunk
; RUN: opt -basicaa -memdep -mldst-motion -S < %s | FileCheck %s
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

%struct.node = type { i32, %struct.node*, %struct.node*, %struct.node*, i32, i32, i32, i32 }

; Function Attrs: nounwind uwtable
define void @sink_store(%struct.node* nocapture %r, i32 %index) {
entry:
  %node.0.in16 = getelementptr inbounds %struct.node, %struct.node* %r, i64 0, i32 2
  %node.017 = load %struct.node*, %struct.node** %node.0.in16, align 8
  %index.addr = alloca i32, align 4
  store i32 %index, i32* %index.addr, align 4
  %0 = load i32, i32* %index.addr, align 4
  %cmp = icmp slt i32 %0, 0
  br i1 %cmp, label %if.then, label %if.else

; CHECK: if.then
if.then:                                          ; preds = %entry
  %1 = load i32, i32* %index.addr, align 4
  %p1 = getelementptr inbounds %struct.node, %struct.node* %node.017, i32 0, i32 6
  ; CHECK: store i32
  store i32 %1, i32* %p1, align 4
  %p2 = getelementptr inbounds %struct.node, %struct.node* %node.017, i32 0, i32 6
  ; CHECK: load i32, i32*
  %barrier = load i32 , i32 * %p2, align 4
  br label %if.end

; CHECK: if.else
if.else:                                          ; preds = %entry
  %2 = load i32, i32* %index.addr, align 4
  %add = add nsw i32 %2, 1
  %p3 = getelementptr inbounds %struct.node, %struct.node* %node.017, i32 0, i32 6
  ; CHECK: store i32
  store i32 %add, i32* %p3, align 4
  br label %if.end

; CHECK: if.end
if.end:                                           ; preds = %if.else, %if.then
; CHECK-NOT: store
  ret void
}
