; RUN: llc %s -o - -stop-after=finalize-isel \
; RUN: | FileCheck %s --implicit-check-not=DBG


; RUN: llc --try-experimental-debuginfo-iterators %s -o - -stop-after=finalize-isel \
; RUN: | FileCheck %s --implicit-check-not=DBG

;; Hand written. Check that we fall back to emitting a list of defs for
;; variables with split allocas (i.e. we want to see DBG_VALUEs and no
;; debug-info-variable entry in the stack slot table).

; CHECK: stack:
; CHECK:   - { id: 0, name: a, type: default, offset: 0, size: 4, alignment: 4, 
; CHECK:       stack-id: default, callee-saved-register: '', callee-saved-restored: true, 
; CHECK:       debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }
; CHECK:   - { id: 1, name: c, type: default, offset: 0, size: 4, alignment: 4, 
; CHECK:       stack-id: default, callee-saved-register: '', callee-saved-restored: true, 
; CHECK:       debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }
; CHECK: DBG_VALUE %stack.0.a, $noreg, !{{.*}}, !DIExpression(DW_OP_deref, DW_OP_LLVM_fragment, 0, 32) 
; CHECK: DBG_VALUE %stack.1.c, $noreg, !{{.*}}, !DIExpression(DW_OP_deref, DW_OP_LLVM_fragment, 64, 32)

target triple = "x86_64-unknown-linux-gnu"

define dso_local void @fun() !dbg !7 {
entry:
  %a = alloca i32, align 4, !DIAssignID !16
  call void @llvm.dbg.assign(metadata i1 undef, metadata !11, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32), metadata !16, metadata ptr %a, metadata !DIExpression()), !dbg !17
  %c = alloca i32, align 4, !DIAssignID !20
  call void @llvm.dbg.assign(metadata i1 undef, metadata !11, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 32), metadata !20, metadata ptr %c, metadata !DIExpression()), !dbg !17
  store i32 5, ptr %a, !DIAssignID !21
  ret void, !dbg !19
}

declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !1000}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 1}
!6 = !{!"clang version 14.0.0"}
!7 = distinct !DISubprogram(name: "fun", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{!11}
!11 = !DILocalVariable(name: "local", scope: !7, file: !1, line: 2, type: !12)
!12 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 96, elements: !14)
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{!15}
!15 = !DISubrange(count: 3)
!16 = distinct !DIAssignID()
!17 = !DILocation(line: 0, scope: !7)
!18 = !DILocation(line: 2, column: 3, scope: !7)
!19 = !DILocation(line: 3, column: 1, scope: !7)
!20 = distinct !DIAssignID()
!21 = distinct !DIAssignID()
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
