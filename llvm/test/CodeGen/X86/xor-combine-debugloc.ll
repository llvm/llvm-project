; RUN: llc -stop-after=expand-isel-pseudos < %s | FileCheck %s
;
; Make sure that when the entry block of IR below is lowered, an instruction
; that implictly defines $eflags has a same debug location with the icmp
; instruction, and the branch instructions have a same debug location with the
; br instruction.
;
; CHECK:      [[DLOC1:![0-9]+]] = !DILocation(line: 5, column: 9, scope: !{{[0-9]+}})
; CHECK:      [[DLOC2:![0-9]+]] = !DILocation(line: 5, column: 7, scope: !{{[0-9]+}})
; CHECK-DAG:  [[VREG1:%[^ ]+]]:gr32 = COPY $esi
; CHECK-DAG:  [[VREG2:%[^ ]+]]:gr32 = COPY $edi
; CHECK:      SUB32rr [[VREG2]], [[VREG1]], implicit-def $eflags, debug-location [[DLOC1]]
; CHECK-NEXT: JE_1{{.*}} implicit $eflags, debug-location [[DLOC2]]
; CHECK-NEXT: JMP_1{{.*}} debug-location [[DLOC2]]

target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define i32 @foo(i32 %x, i32 %y) !dbg !4 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %x, i64 0, metadata !9, metadata !11), !dbg !12
  tail call void @llvm.dbg.value(metadata i32 %y, i64 0, metadata !10, metadata !11), !dbg !13
  %cmp = icmp ne i32 %x, %y, !dbg !14
  br i1 %cmp, label %if.then, label %if.else, !dbg !16

if.then:                                          ; preds = %entry
  %call = tail call i32 (...) @bar() #3, !dbg !17
  br label %return, !dbg !18

if.else:                                          ; preds = %entry
  %call1 = tail call i32 (...) @baz() #3, !dbg !19
  br label %return, !dbg !20

return:                                           ; preds = %if.else, %if.then
  %retval.0 = phi i32 [ %call, %if.then ], [ %call1, %if.else ]
  ret i32 %retval.0, !dbg !21
}

declare i32 @bar(...)
declare i32 @baz(...)

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "foo.c", directory: "b/")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 4, type: !5, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !8)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{!9, !10}
!9 = !DILocalVariable(name: "x", arg: 1, scope: !4, file: !1, line: 4, type: !7)
!10 = !DILocalVariable(name: "y", arg: 2, scope: !4, file: !1, line: 4, type: !7)
!11 = !DIExpression()
!12 = !DILocation(line: 4, column: 13, scope: !4)
!13 = !DILocation(line: 4, column: 20, scope: !4)
!14 = !DILocation(line: 5, column: 9, scope: !15)
!15 = distinct !DILexicalBlock(scope: !4, file: !1, line: 5, column: 7)
!16 = !DILocation(line: 5, column: 7, scope: !4)
!17 = !DILocation(line: 6, column: 12, scope: !15)
!18 = !DILocation(line: 6, column: 5, scope: !15)
!19 = !DILocation(line: 8, column: 12, scope: !15)
!20 = !DILocation(line: 8, column: 5, scope: !15)
!21 = !DILocation(line: 9, column: 1, scope: !4)
