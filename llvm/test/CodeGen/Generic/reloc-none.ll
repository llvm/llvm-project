; RUN: llc < %s | FileCheck %s

; CHECK: .reloc {{.*}}, BFD_RELOC_NONE, foo

%1 = type opaque
@foo = external global %1

define void @test_reloc_none() {
  call void @llvm.reloc.none(ptr @foo)
  ret void
}

declare void @llvm.reloc.none(ptr)
