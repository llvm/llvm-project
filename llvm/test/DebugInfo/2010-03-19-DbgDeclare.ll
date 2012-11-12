; RUN: llvm-as < %s | opt -verify -disable-output

define void @Foo(i32 %a, i32 %b) {
entry:
  call void @llvm.dbg.declare(metadata !{i32* null}, metadata !1)
  ret void
}

!0 = metadata !{i32 662302, i32 26, metadata !1, null}
!1 = metadata !{i32 4, metadata !"foo"}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone
