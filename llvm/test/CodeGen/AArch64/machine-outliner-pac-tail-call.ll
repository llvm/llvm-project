; RUN: llc -mtriple=aarch64-linux-android -enable-machine-outliner < %s | FileCheck %s

; Check that the MachineOutliner successfully outlines a sequence ending in a tail call
; when PAC-RET (Pointer Authentication) is enabled.
; Fixes #230519.

; CHECK-LABEL: f1:
; CHECK: bl OUTLINED_FUNCTION_0
; CHECK-LABEL: f2:
; CHECK: bl OUTLINED_FUNCTION_0

; CHECK-LABEL: OUTLINED_FUNCTION_0:
; CHECK: b sink

declare i32 @sink(ptr, i32, i32, i32)

define i32 @f1(ptr %p) "sign-return-address"="all" minsize {
entry:
  %arrayidx = getelementptr inbounds i32, ptr %p, i64 1
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, 3
  %xor = xor i32 %add, 85
  store i32 %xor, ptr %p, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr %p, i64 3
  %1 = load i32, ptr %arrayidx1, align 4
  %mul = mul nsw i32 %1, 5
  %add2 = add nsw i32 %mul, 17
  %arrayidx3 = getelementptr inbounds i32, ptr %p, i64 2
  store i32 %add2, ptr %arrayidx3, align 4
  %arrayidx4 = getelementptr inbounds i32, ptr %p, i64 5
  %2 = load i32, ptr %arrayidx4, align 4
  %sub = add nsw i32 %2, -7
  %xor5 = xor i32 %sub, 34
  %arrayidx6 = getelementptr inbounds i32, ptr %p, i64 4
  store i32 %xor5, ptr %arrayidx6, align 4
  %call = tail call i32 @sink(ptr %p, i32 %xor, i32 %add2, i32 %xor5)
  %add7 = add nsw i32 %call, 1
  ret i32 %add7
}

define i32 @f2(ptr %p) "sign-return-address"="all" minsize {
entry:
  %arrayidx = getelementptr inbounds i32, ptr %p, i64 1
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, 3
  %xor = xor i32 %add, 85
  store i32 %xor, ptr %p, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr %p, i64 3
  %1 = load i32, ptr %arrayidx1, align 4
  %mul = mul nsw i32 %1, 5
  %add2 = add nsw i32 %mul, 17
  %arrayidx3 = getelementptr inbounds i32, ptr %p, i64 2
  store i32 %add2, ptr %arrayidx3, align 4
  %arrayidx4 = getelementptr inbounds i32, ptr %p, i64 5
  %2 = load i32, ptr %arrayidx4, align 4
  %sub = add nsw i32 %2, -7
  %xor5 = xor i32 %sub, 34
  %arrayidx6 = getelementptr inbounds i32, ptr %p, i64 4
  store i32 %xor5, ptr %arrayidx6, align 4
  %call = tail call i32 @sink(ptr %p, i32 %xor, i32 %add2, i32 %xor5)
  %add7 = add nsw i32 %call, 2
  ret i32 %add7
}
