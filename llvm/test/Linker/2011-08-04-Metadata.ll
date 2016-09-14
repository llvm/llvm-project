; RUN: llvm-link %s %p/2011-08-04-Metadata2.ll -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s
; Test if internal global variable's debug info is merged appropriately or not.

; CHECK: @x = internal global i32 0, align 4, !dbg [[DI1:![0-9]+]]
; CHECK: @x.1 = internal global i32 0, align 4, !dbg [[DI2:![0-9]+]]

; CHECK: [[DI1]] = !DIGlobalVariable(name: "x",
; CHECK-NOT:               linkageName:
; CHECK: [[DI2]] = !DIGlobalVariable(name: "x",
; CHECK-NOT:               linkageName:
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

@x = internal global i32 0, align 4, !dbg !5

define void @foo() nounwind uwtable ssp !dbg !1 {
entry:
  store i32 1, i32* @x, align 4, !dbg !7
  ret void, !dbg !7
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.0 ()", isOptimized: true, emissionKind: FullDebug, file: !9, enums: !{}, retainedTypes: !{}, globals: !{!5})
!1 = distinct !DISubprogram(name: "foo", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !0, file: !9, scope: !2, type: !3)
!2 = !DIFile(filename: "/tmp/one.c", directory: "/Volumes/Lalgate/Slate/D")
!3 = !DISubroutineType(types: !4)
!4 = !{null}
!5 = !DIGlobalVariable(name: "x", line: 2, isLocal: true, isDefinition: true, scope: !0, file: !2, type: !6)
!6 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!7 = !DILocation(line: 3, column: 14, scope: !8)
!8 = distinct !DILexicalBlock(line: 3, column: 12, file: !9, scope: !1)
!9 = !DIFile(filename: "/tmp/one.c", directory: "/Volumes/Lalgate/Slate/D")
!11 = !{i32 1, !"Debug Info Version", i32 3}
