; RUN: opt < %s -passes='logical-sroa' -S -debug | FileCheck %s --check-prefixes=CHECK

define i32 @test_normal_gep() {
; CHECK-LABEL: @test_normal_gep(
; CHECK-NEXT:  entry:
entry:
  %tmp = call elementtype({ i32, i32 }) ptr @llvm.structured.alloca.p0()
  %0 = getelementptr i8, ptr %tmp, i32 0
  %1 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype({ i32, i32 }) %tmp, i32 1)
; CHECK:  %tmp = call elementtype({ i32, i32 }) ptr @llvm.structured.alloca.p0()
; CHECK:  %[[#a:]] = getelementptr i8, ptr %tmp, i32 0
; CHECK:  %[[#b:]] = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype({ i32, i32 }) %tmp, i32 1)

  store i32 0, ptr %0
  store i32 1, ptr %1
  %a = load i32, ptr %0
  %b = load i32, ptr %1
; CHECK:  store i32 0, ptr %[[#a]]
; CHECK:  store i32 1, ptr %[[#b]]
; CHECK:  %a = load i32, ptr %[[#a]]
; CHECK:  %b = load i32, ptr %[[#b]]

  %c = add i32 %a, %b
  ret i32 %c
}
