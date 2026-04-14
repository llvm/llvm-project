; RUN: opt < %s -passes='lsroa' -S -debug | FileCheck %s --check-prefixes=CHECK

declare void @llvm.lifetime.start.p0(ptr nocapture)
declare void @llvm.lifetime.end.p0(ptr nocapture)
declare ptr @llvm.structured.alloca.p0()
declare ptr @llvm.structured.gep.p0(ptr, ...)

define i32 @test_simple_array() {
; CHECK-LABEL: @test_simple_array(
; CHECK-NEXT:  entry:
entry:
  %tmp = call elementtype([10 x i32]) ptr @llvm.structured.alloca.p0()
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([10 x i32]) %tmp, i32 0)
  store i32 0, ptr %ptr
  %res = load i32, ptr %ptr
  ret i32 %res

; CHECK-NEXT:  %tmp = call elementtype([10 x i32]) ptr @llvm.structured.alloca.p0()
; CHECK-NEXT:  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([10 x i32]) %tmp, i32 0)
; CHECK-NEXT:  store i32 0, ptr %ptr
; CHECK-NEXT:  %res = load i32, ptr %ptr
; CHECK-NEXT:  ret i32 %res
}

