; RUN: opt -inline -S -o - < %s | FileCheck %s
; Check that debug info is stripped when inlining into a nodebug function.

declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

define void @foo() !dbg !2 {
entry:
  %a = alloca i32
  call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !3, metadata !DIExpression()), !dbg !6
  store i32 0, i32* %a, !dbg !6
  ret void, !dbg !6
}

; CHECK: define void @bar()
define void @bar() {
; CHECK-NEXT: entry
entry:
; CHECK-NEXT: alloca i32
; CHECK-NOT: dbg
; CHECK: ret void
  call void @foo()
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", emissionKind: FullDebug)
!1 = !DIFile(filename: "x.c", directory: "/")
!2 = distinct !DISubprogram(name: "foo", scope: !0, isDefinition: true, unit: !0)
!3 = !DILocalVariable(name: "a", arg: 1, scope: !2, file: !1, line: 1, type: !5)
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !DILocation(line: 1, scope: !2)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 1, !"Debug Info Version", i32 3}
