; RUN: llvm-as < %s | opt -verify -S -asm-verbose | FileCheck %s

; CHECK: lang 0x8001

define void @Foo(i32 %a, i32 %b) {
entry:
  call void @llvm.dbg.declare(metadata !{i32* null}, metadata !1)
  ret void
}
!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!5}
!2 = metadata !{i32 786449, metadata !4, i32 32769, metadata !"clang version 3.3 ", i1 false, metadata !"", i32 0, metadata !3, metadata !3, metadata !3, metadata !3,  metadata !3, metadata !""} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/blaikie/dev/scratch/scratch.cpp] [lang 0x8001]
!3 = metadata !{}
!0 = metadata !{i32 662302, i32 26, metadata !1, null}
!1 = metadata !{i32 4, metadata !"foo"}
!4 = metadata !{metadata !"scratch.cpp", metadata !"/usr/local/google/home/blaikie/dev/scratch"}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone
!5 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
