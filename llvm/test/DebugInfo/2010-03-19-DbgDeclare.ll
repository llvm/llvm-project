; RUN: opt < %s -verify -S | FileCheck %s

; CHECK: [DW_LANG_Mips_Assembler]

define void @Foo(i32 %a, i32 %b) {
entry:
  call void @llvm.dbg.declare(metadata i32* null, metadata !1, metadata !{!"0x102"})
  ret void
}
!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!5}
!2 = !{!"0x11\0032769\00clang version 3.3 \000\00\000\00\001", !4, !3, !3, !3, !3,  !3} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/blaikie/dev/scratch/scratch.cpp] [lang 0x8001]
!3 = !{}
!0 = !MDLocation(line: 662302, column: 26, scope: !1)
!1 = !{i32 4, !"foo"}
!4 = !{!"scratch.cpp", !"/usr/local/google/home/blaikie/dev/scratch"}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone
!5 = !{i32 1, !"Debug Info Version", i32 2}
