; RUN: not llvm-as -disable-output 2>&1 %s | FileCheck %s

; CHECK: llvm.reloc.none argument must be a metadata string
; CHECK-NEXT: call void @llvm.reloc.none(metadata !0)

define void @test_reloc_none_bad_arg() {
  call void @llvm.reloc.none(metadata !0)
  ret void
}

declare void @llvm.reloc.none(metadata)

!0 = !{}
