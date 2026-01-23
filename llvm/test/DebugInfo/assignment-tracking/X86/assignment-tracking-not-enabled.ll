; RUN: llc %s -stop-after=finalize-isel -o - \
; RUN: | FileCheck %s --implicit-check-not=DBG_

;; Check that SelectionDAG downgrades dbg.assigns to dbg.values if assignment
;; tracking isn't enabled (e.g. if the module flag
;; "debug-info-assignment-tracking" is missing / false).

;; With assignment tracking enabled we'd see the variable put into the stack
;; slot side table because the variable is always located in its stack
;; slot. Check there's no debug-info saved there:
;; CHECK: stack:
;; CHECK-NEXT: - { id: 0, name: x, type: default, offset: 0, size: 4, alignment: 4,
;; CHECK-NEXT:     stack-id: default, callee-saved-register: '', callee-saved-restored: true,
;; CHECK-NEXT:     debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }

;; With assignment tracking disabled we should see the dbg.assigns downgraded
;; to dbg.values, which become DBG _VALUEs/_INSTR_REFs.
; CHECK: bb.0.entry:
; CHECK: DBG_VALUE $noreg

; CHECK: bb.1.if.then:
; CHECK: DBG_INSTR_REF

; CHECK: bb.2.if.else:
; CHECK: DBG_VALUE 2

target triple = "x86_64-unknown-unknown"

@g = dso_local global i32 0, align 4, !dbg !0

define dso_local noundef i32 @_Z3funv() #0 !dbg !15 {
entry:
  %x = alloca i32, align 4, !DIAssignID !19
  call void @llvm.dbg.assign(metadata i1 undef, metadata !20, metadata !DIExpression(), metadata !19, metadata ptr %x, metadata !DIExpression()), !dbg !21
  %0 = load i32, ptr @g, align 4
  %tobool = icmp ne i32 %0, 0
  br i1 %tobool, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %call = call noundef i32 @_Z3getv()
  store i32 %call, ptr %x, align 4, !DIAssignID !27
  call void @llvm.dbg.assign(metadata i32 %call, metadata !20, metadata !DIExpression(), metadata !27, metadata ptr %x, metadata !DIExpression()), !dbg !21
  br label %if.end

if.else:                                          ; preds = %entry
  store i32 2, ptr %x, align 4, !DIAssignID !30
  call void @llvm.dbg.assign(metadata i32 2, metadata !20, metadata !DIExpression(), metadata !30, metadata ptr %x, metadata !DIExpression()), !dbg !21
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %1 = load i32, ptr %x, align 4
  ret i32 %1
}

declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata) #1
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1
declare noundef i32 @_Z3getv() #2

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!6, !7}
!llvm.ident = !{!14}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 17.0.0)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.cpp", directory: "/")
!4 = !{!0}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !{i32 7, !"Dwarf Version", i32 5}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{!"clang version 17.0.0"}
!15 = distinct !DISubprogram(name: "fun", linkageName: "_Z3funv", scope: !3, file: !3, line: 3, type: !16, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !18)
!16 = !DISubroutineType(types: !17)
!17 = !{!5}
!18 = !{}
!19 = distinct !DIAssignID()
!20 = !DILocalVariable(name: "x", scope: !15, file: !3, line: 4, type: !5)
!21 = !DILocation(line: 0, scope: !15)
!23 = distinct !DILexicalBlock(scope: !15, file: !3, line: 5, column: 7)
!27 = distinct !DIAssignID()
!30 = distinct !DIAssignID()
