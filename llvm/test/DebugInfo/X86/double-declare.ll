; RUN: llc -mtriple=x86_64-apple-darwin -O0 -filetype=obj -o - < %s | llvm-dwarfdump -v -debug-info - | FileCheck %s
; RUN: llc --try-experimental-debuginfo-iterators -mtriple=x86_64-apple-darwin -O0 -filetype=obj -o - < %s | llvm-dwarfdump -v -debug-info - | FileCheck %s
; PR33157. Don't crash on duplicate dbg.declare.
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_AT_location [DW_FORM_exprloc]
; CHECK-NOT: DW_AT_location
@g = external global i32
@h = external global i32

declare void @llvm.dbg.declare(metadata, metadata, metadata)

define void @f(ptr byval(i32) %p, i1 %c) !dbg !5 {
  br i1 %c, label %x, label %y

x:
  call void @llvm.dbg.declare(metadata ptr %p, metadata !10, metadata !DIExpression()), !dbg !12
  store i32 42, ptr @g, !dbg !12
  br label %done

y:
  call void @llvm.dbg.declare(metadata ptr %p, metadata !10, metadata !DIExpression()), !dbg !12
  store i32 42, ptr @h, !dbg !12
  br label %done

done:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22, !23}

!0 = distinct !DICompileUnit(language: DW_LANG_ObjC, file: !1, producer: "clang version 5.0.0 ", isOptimized: true, runtimeVersion: 2, emissionKind: FullDebug)
!1 = !DIFile(filename: "<stdin>", directory: "C:\5Csrc\5Cllvm-project\5Cbuild")
!5 = distinct !DISubprogram(name: "f", isLocal: true, isDefinition: true, scopeLine: 37, flags: DIFlagPrototyped, isOptimized: true, unit: !0, type: !99, scope: !1)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DILocalVariable(name: "aRect", arg: 1, scope: !11, file: !1, line: 38, type: !6)
!11 = distinct !DILexicalBlock(scope: !98, file: !1, line: 38)
!12 = !DILocation(line: 43, scope: !11, inlinedAt: !13)
!13 = distinct !DILocation(line: 43, scope: !5)
!22 = !{i32 2, !"Dwarf Version", i32 4}
!23 = !{i32 2, !"Debug Info Version", i32 3}
!62 = !{!10}
!98 = distinct !DISubprogram(name: "NSMaxX", scope: !1, file: !1, line: 27, isLocal: true, isDefinition: true, scopeLine: 27, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !62, type: !99)
!99 = !DISubroutineType(types: !100)
!100 = !{null}
