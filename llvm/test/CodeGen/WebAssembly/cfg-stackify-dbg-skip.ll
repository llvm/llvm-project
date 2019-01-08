; RUN: llc %s -stop-after wasm-cfg-stackify -o - | FileCheck %s

; The test ensures "block" instruction is not inserted in the middle of a group
; of instructions that form a stackified expression when DBG_VALUE is present
; among them.

; CHECK: body:
; CHECK: BLOCK
;                       <-- Stackified expression starts
; CHECK-NEXT: LOCAL_GET_I64
; CHECK-NEXT: I32_WRAP_I64
; CHECK-NEXT: DBG_VALUE
;                       <-- BLOCK should NOT be placed here!
; CHECK-NEXT: BR_UNLESS
;                       <-- Stackified expression ends

target triple = "wasm32-unknown-unknown"

define void @foo(i64 %arg) {
start:
  %val = trunc i64 %arg to i32
  %cmp = icmp eq i32 %val, 0
  call void @llvm.dbg.value(metadata i32 %val, metadata !46, metadata !DIExpression()), !dbg !105
  br i1 %cmp, label %bb2, label %bb1
bb1:                                              ; preds = %start
  call void @bar()
  br label %bb2
bb2:                                              ; preds = %bb1, start
  ret void
}

declare void @bar()
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!33}
!0 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !6, producer: "clang LLVM (rustc version 1.30.0-dev)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !2)
!2 = !{}
!6 = !DIFile(filename: "<unknown>", directory: "")
!22 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "&str", file: !6, size: 64, align: 32, elements: !{}, identifier: "111094d970b097647de579f9c509ef08")
!33 = !{i32 2, !"Debug Info Version", i32 3}
!35 = distinct !DILexicalBlock(scope: !37, file: !6, line: 357, column: 8)
!37 = distinct !DISubprogram(name: "foobar", linkageName: "_fooba", scope: !38, file: !6, line: 353, type: !39, isLocal: true, isDefinition: true, scopeLine: 353, flags: DIFlagPrototyped, isOptimized: true, unit: !0, templateParams: !2, retainedNodes: !42)
!38 = !DINamespace(name: "ptr", scope: null)
!39 = !DISubroutineType(types: !2)
!42 = !{!46}
!46 = !DILocalVariable(name: "z", scope: !35, file: !6, line: 357, type: !22, align: 4)
!105 = !DILocation(line: 357, column: 12, scope: !35)
