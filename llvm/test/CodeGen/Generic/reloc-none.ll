; RUN: llc < %s | FileCheck %s

; CHECK: .reloc {{.*}}, BFD_RELOC_NONE, foo

define void @test_reloc_none() {
  call void @llvm.reloc.none(metadata !"foo")
  ret void
}

declare void @llvm.reloc.none(metadata)
