; RUN: opt < %s -argpromotion -S | FileCheck %s
; CHECK: call void @test(i32 %
; CHECK: void (i32)* @test, {{.*}} ; [ DW_TAG_subprogram ] {{.*}} [test]

declare void @sink(i32)

define internal void @test(i32** %X) {
  %1 = load i32** %X, align 8
  %2 = load i32* %1, align 8
  call void @sink(i32 %2)
  ret void
}

define void @caller(i32** %Y) {
  call void @test(i32** %Y)
  ret void
}

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!3}

!0 = !{i32 2, !"Debug Info Version", i32 2}
!1 = !MDLocation(line: 8, scope: !2)
!2 = !{!"0x2e\00test\00test\00\003\001\001\000\006\00256\000\003", null, null, null, null, void (i32**)* @test, null, null, null} ; [ DW_TAG_subprogram ]
!3 = !{!"0x11\004\00clang version 3.5.0 \000\00\000\00\002", null, null, null, !4, null, null} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/blaikie/dev/scratch/pr20038/reduce/<stdin>] [DW_LANG_C_plus_plus]
!4 = !{!2}
