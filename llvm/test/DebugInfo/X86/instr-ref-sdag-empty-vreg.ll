; RUN: llc %s -o - -stop-before=finalize-isel \
; RUN:  -experimental-debug-variable-locations \
; RUN: | FileCheck %s

;; The multiply by zero of %width below causes %width to become unused very late
;; in SelectionDAG. As a result, DBG_INSTR_REFs are produced that refer to a
;; vreg that is never defined, which risks a crash. Check that we don't crash,
;; and produce an empty variable location.

; CHECK: DBG_VALUE_LIST {{.+}}, $noreg
; CHECK: DBG_VALUE_LIST {{.+}}, $noreg, 4

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

%class.Color = type { i8 }

define hidden void @_Z14drawXZWideLineR4Vec3RK5Colorf(ptr %color, float %width) local_unnamed_addr !dbg !7 {
cond.false.i:
  br label %_ZN4Vec39normalizeEv.exit, !dbg !12

_ZN4Vec39normalizeEv.exit:                        ; preds = %cond.false.i
  call void @llvm.dbg.value(metadata float %width, metadata !11, metadata !DIExpression()), !dbg !12
  call void @llvm.dbg.value(metadata !DIArgList(float %width, i32 4), metadata !11, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_mul, DW_OP_stack_value)), !dbg !12
  %mul.i = fmul float %width, 0.000000e+00, !dbg !12
  ret void, !dbg !12
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "foo.cpp", directory: ".")
!2 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_signed)
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: !1, file: !1, line: 6, type: !8, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{!2, !2}
!10 = !{!11}
!11 = !DILocalVariable(name: "baz", scope: !7, file: !1, line: 7, type: !2)
!12 = !DILocation(line: 10, scope: !7)
