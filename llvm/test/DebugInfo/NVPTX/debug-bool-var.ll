; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda -print-after=finalize-isel -o /dev/null 2>&1 | FileCheck %s

declare void @foo(i32)

define void @test1(i32 noundef %gid) !dbg !3 {
entry:
  ;
  ; Equivalent of code:
  ;   extern void foo(int);
  ;   void test_kernel_bool(int a) {
  ;     bool xyz = a == 0;
  ;     foo(xyz);
  ;   }
  ;
  ; CHECK-LABEL: Machine code for function test1
  ; CHECK: DBG_VALUE %[[#]]:int32regs, $noreg, !"xyz", !DIExpression(), debug-location ![[#]]; test.cu:2 line no:6
  ;
  %cmp = icmp eq i32 %gid, 0, !dbg !12
  %conv = zext i1 %cmp to i32, !dbg !12
  %conv1 = trunc i32 %conv to i8, !dbg !12
    #dbg_value(i8 %conv1, !10, !DIExpression(), !13)
  %conv3 = sext i8 %conv1 to i32
  call void @foo(i32 %conv3)
  ret void
}

define void @test2(i32 noundef %gid) !dbg !14 {
entry:
  ;
  ; Equivalent of code:
  ;   extern void foo(int);
  ;   void test_kernel_bool(int a) {
  ;     unsigned char abc = a == 0;
  ;     foo(abc);
  ;   }
  ;
  ; CHECK-LABEL: Machine code for function test2
  ; CHECK: DBG_VALUE %[[#]]:int32regs, $noreg, !"abc", !DIExpression(), debug-location ![[#]]; test.cu:12 line no:11
  ;
  %cmp = icmp eq i32 %gid, 0, !dbg !17
  %conv = zext i1 %cmp to i32, !dbg !17
  %conv1 = trunc i32 %conv to i8, !dbg !17
    #dbg_value(i8 %conv1, !16, !DIExpression(), !18)
  %conv3 = zext i8 %conv1 to i32
  call void @foo(i32 %conv3)
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cu", directory: "/source/dir")
!2 = !{i32 1, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "test1", linkageName: "_test1i", scope: !1, file: !1, line: 5, type: !4, scopeLine: 5, unit: !0, retainedNodes: !8)
!4 = !DISubroutineType(types: !5)
!5 = !{!6, !7}
!6 = !DIBasicType(tag: DW_TAG_unspecified_type, name: "void")
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{}
!9 = distinct !DILexicalBlock(scope: !3, file: !1, line: 5, column: 30)
!10 = !DILocalVariable(name: "xyz", scope: !9, file: !1, line: 6, type: !11)
!11 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!12 = !DILocation(line: 1, column: 3, scope: !9)
!13 = !DILocation(line: 2, scope: !9)
!14 = distinct !DISubprogram(name: "test2", linkageName: "_test2i", scope: !1, file: !1, line: 10, type: !4, scopeLine: 10, unit: !0, retainedNodes: !8)
!15 = distinct !DILexicalBlock(scope: !14, file: !1, line: 10, column: 30)
!16 = !DILocalVariable(name: "abc", scope: !15, file: !1, line: 11, type: !11)
!17 = !DILocation(line: 11, column: 3, scope: !15)
!18 = !DILocation(line: 12, scope: !15)
