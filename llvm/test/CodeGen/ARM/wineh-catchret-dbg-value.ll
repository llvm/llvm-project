; LiveDebugValues puts a DBG_VALUE between the catch funclet's epilogue and its
; CATCHRET. The continuation load in r0 still goes before the epilogue.

; RUN: llc -mtriple=thumbv7-windows-msvc -O2 < %s | FileCheck %s
; RUN: llc -mtriple=thumbv7-windows-msvc -O2 -filetype=obj < %s -o /dev/null

; CHECK-LABEL: "?catch$
; CHECK:       movw r0, :lower16:[[CONT:\$MBB[0-9_]+]]
; CHECK-NEXT:  movt r0, :upper16:[[CONT]]
; CHECK-NEXT:  .seh_startepilogue

declare void @_CxxThrowException(ptr, ptr)
declare i32 @__CxxFrameHandler3(...)

define void @f(i1 %c) personality ptr @__CxxFrameHandler3 !dbg !4 {
entry:
  br i1 %c, label %ret, label %throw

throw:
  invoke void @_CxxThrowException(ptr null, ptr null)
          to label %unreachable unwind label %dispatch

unreachable:
  unreachable

dispatch:
  %cs = catchswitch within none [label %catch] unwind to caller

catch:
  %p = catchpad within %cs [ptr null, i32 0, ptr null]
  catchret from %p to label %cont

cont:
    #dbg_value(ptr null, !6, !DIExpression(DW_OP_LLVM_fragment, 0, 32), !8)
  call void @llvm.lifetime.end.p0(ptr poison)
  br label %ret

ret:
  ret void
}

declare void @llvm.lifetime.end.p0(ptr captures(none))

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.cpp", directory: "/")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DISubroutineType(types: !{})
!6 = !DILocalVariable(name: "x", scope: !4, file: !1, line: 2, type: !7)
!7 = !DICompositeType(tag: DW_TAG_class_type, name: "X", file: !1, line: 1, size: 64, elements: !{})
!8 = !DILocation(line: 0, scope: !4)
