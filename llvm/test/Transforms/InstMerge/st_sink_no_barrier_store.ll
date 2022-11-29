; Test to make sure that stores in a diamond get merged with a non barrier store after the store instruction to be sunk
; Stores sunks into the footer.
; RUN: opt -mldst-motion -S < %s | FileCheck %s
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

%struct.node = type { i32, ptr, ptr, ptr, i32, i32, i32, i32 }

; Function Attrs: nounwind uwtable
define void @sink_store(ptr nocapture %r, i32 %index) {
entry:
  %node.0.in16 = getelementptr inbounds %struct.node, ptr %r, i64 0, i32 2
  %node.017 = load ptr, ptr %node.0.in16, align 8
  %index.addr = alloca i32, align 4
  store i32 %index, ptr %index.addr, align 4
  %0 = load i32, ptr %index.addr, align 4
  %cmp = icmp slt i32 %0, 0
  br i1 %cmp, label %if.then, label %if.else

; CHECK: if.then
if.then:                                          ; preds = %entry
  %1 = load i32, ptr %index.addr, align 4
  %p1 = getelementptr inbounds %struct.node, ptr %node.017, i32 0, i32 6
  ; CHECK-NOT: store i32
  store i32 %1, ptr %p1, align 4
  br label %if.end

; CHECK: if.else
if.else:                                          ; preds = %entry
  %2 = load i32, ptr %index.addr, align 4
  %add = add nsw i32 %2, 1
  %p2 = getelementptr inbounds %struct.node, ptr %node.017, i32 0, i32 6
  store i32 %add, ptr %p2, align 4
  %p3 = getelementptr inbounds %struct.node, ptr %node.017, i32 5, i32 6
  ; CHECK: store i32
  store i32 %add, ptr %p3, align 4  			  ; This is not a barrier
  br label %if.end

; CHECK: if.end
if.end:                                           ; preds = %if.else, %if.then
; CHECK: store
  ret void
}
