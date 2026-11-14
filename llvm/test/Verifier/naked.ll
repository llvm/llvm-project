; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

; CHECK: cannot use argument of naked function
define void @test(ptr %ptr) naked {
  getelementptr i8, ptr %ptr, i64 1
  call void @llvm.trap()
  unreachable
}
