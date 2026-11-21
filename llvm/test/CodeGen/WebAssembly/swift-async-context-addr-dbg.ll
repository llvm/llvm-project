; RUN: llc < %s -stop-after=wasm-explicit-locals | FileCheck %s

; After an await resume funclet stores its async context to local 0, debug
; values of the swiftasync argument cannot refer to local 0 anymore.

target triple = "wasm32-unknown-unknown"

declare swiftcc void @swift_task_dealloc(ptr)
declare ptr @llvm.swift.async.context.addr()

; The swiftasync argument is not read after the store: its debug locations end
; at the store.
; CHECK-LABEL: name: undef_after_store
; CHECK:      DBG_VALUE target-index(wasm-local), $noreg, ![[X:[0-9]+]], !DIExpression(DW_OP_deref)
; CHECK:      LOCAL_SET_I32 0,
; CHECK-NEXT: DBG_VALUE $noreg, $noreg, ![[X]], !DIExpression(DW_OP_deref)
; CHECK-NEXT: DBG_VALUE $noreg, $noreg, ![[X]], !DIExpression(DW_OP_deref, DW_OP_deref)
define swiftcc void @undef_after_store(ptr swiftasync %callee_ctx) !dbg !5 {
    #dbg_value(ptr %callee_ctx, !10, !DIExpression(DW_OP_deref), !11)
  %ctx = load ptr, ptr %callee_ctx, !dbg !11
  %addr = call ptr @llvm.swift.async.context.addr(), !dbg !11
  store ptr %ctx, ptr %addr, !dbg !11
    #dbg_value(ptr %callee_ctx, !10, !DIExpression(DW_OP_deref, DW_OP_deref), !11)
  call swiftcc void @swift_task_dealloc(ptr %ctx), !dbg !11
  ret void, !dbg !11
}

; The swiftasync argument is read after the store, so it moves to another
; local, which its debug values refer to.
; CHECK-LABEL: name: copied
; CHECK:      %[[TMP:[0-9]+]]:i32 = LOCAL_GET_I32 0
; CHECK-NEXT: LOCAL_SET_I32 [[COPY:[0-9]+]], %[[TMP]]
; CHECK-NEXT: DBG_VALUE target-index(wasm-local) + [[COPY]], $noreg, ![[Y:[0-9]+]], !DIExpression(DW_OP_deref)
; CHECK:      LOCAL_SET_I32 0,
; CHECK-NEXT: DBG_VALUE target-index(wasm-local) + [[COPY]], $noreg, ![[Y]], !DIExpression(DW_OP_deref, DW_OP_deref)
define swiftcc void @copied(ptr swiftasync %callee_ctx) !dbg !12 {
    #dbg_value(ptr %callee_ctx, !13, !DIExpression(DW_OP_deref), !14)
  %ctx = load ptr, ptr %callee_ctx, !dbg !14
  %addr = call ptr @llvm.swift.async.context.addr(), !dbg !14
  store ptr %ctx, ptr %addr, !dbg !14
    #dbg_value(ptr %callee_ctx, !13, !DIExpression(DW_OP_deref, DW_OP_deref), !14)
  %again = load ptr, ptr %callee_ctx, !dbg !14
  call swiftcc void @swift_task_dealloc(ptr %again), !dbg !14
  ret void, !dbg !14
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!0 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !1, producer: "t", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.swift", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "undef_after_store", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !{}
!9 = !DIBasicType(name: "Int", size: 32, encoding: DW_ATE_signed)
!10 = !DILocalVariable(name: "x", scope: !5, file: !1, line: 1, type: !9)
!11 = !DILocation(line: 1, column: 1, scope: !5)
!12 = distinct !DISubprogram(name: "copied", scope: !1, file: !1, line: 2, type: !6, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
!13 = !DILocalVariable(name: "y", scope: !12, file: !1, line: 2, type: !9)
!14 = !DILocation(line: 2, column: 1, scope: !12)
