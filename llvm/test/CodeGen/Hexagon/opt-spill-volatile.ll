; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; Check that the load/store to the volatile stack object has not been
; optimized away.

target triple = "hexagon"

; CHECK-LABEL: foo
; CHECK: memw(r29+#4) =
; CHECK: = memw(r29+#4)
define i32 @foo(i32 %a) #0 {
entry:
  %x = alloca i32, align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr %x)
  store volatile i32 0, ptr %x, align 4
  %call = tail call i32 @bar() #0
  %x.0.x.0. = load volatile i32, ptr %x, align 4
  %add = add nsw i32 %x.0.x.0., %a
  call void @llvm.lifetime.end.p0(i64 4, ptr %x)
  ret i32 %add
}

declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #1
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #1

declare i32 @bar(...) #0

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }
