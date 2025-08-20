; RUN: not llvm-as -disable-output 2>&1 %s | FileCheck %s

; CHECK: llvm.reloc.none argument must be a global value
; CHECK-NEXT: call void @llvm.reloc.none(ptr %foo)

define void @test_reloc_none_bad_arg(ptr %foo) {
  call void @llvm.reloc.none(ptr %foo)
  ret void
}

declare void @llvm.reloc.none(ptr)

!0 = !{}
