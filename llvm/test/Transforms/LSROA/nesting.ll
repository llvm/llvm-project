; RUN: opt < %s -passes='logical-sroa' -S | FileCheck %s --check-prefixes=CHECK

declare void @llvm.lifetime.start.p0(ptr nocapture)
declare void @llvm.lifetime.end.p0(ptr nocapture)
declare ptr @llvm.structured.alloca.p0()
declare ptr @llvm.structured.gep.p0(ptr, ...)

%S = type { i32, { i32, i32 } }

define i32 @test_nested_struct() {
; CHECK-LABEL: @test_nested_struct(
; CHECK-NEXT:  entry:
entry:
  %tmp = call elementtype(%S) ptr @llvm.structured.alloca.p0()
  %0 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%S) %tmp, i32 0)
  %1 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%S) %tmp, i32 1, i32 0)

; CHECK: %[[#a:]] = call elementtype(i32) ptr @llvm.structured.alloca.p0()
; CHECK: %[[#b:]] = call elementtype({ i32, i32 }) ptr @llvm.structured.alloca.p0()
; CHECK: %[[#ptr:]] = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype({ i32, i32 }) %[[#b]], i32 0)

  store i32 0, ptr %0
  store i32 1, ptr %1
  %a = load i32, ptr %0
  %b = load i32, ptr %1
; CHECK:  store i32 0, ptr %[[#a]]
; CHECK:  store i32 1, ptr %[[#ptr]]
; CHECK:  %a = load i32, ptr %[[#a]]
; CHECK:  %b = load i32, ptr %[[#ptr]]

  %c = add i32 %a, %b
  ret i32 %c
}
