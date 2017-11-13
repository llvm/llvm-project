; RUN: llc -filetype=asm %s -o -  -stop-after=livedebugvalues | FileCheck %s
; This tests that transferring debug info describing the lower bits of
; an extended SDNode works.
target triple = "thumbv6m-apple-unknown-macho"
define arm_aapcscc i64 @f(double %a) !dbg !5 {
entry:
  %0 = bitcast double %a to i64
  %extract.t84 = trunc i64 %0 to i32
  tail call void @llvm.dbg.value(metadata i32 %extract.t84, metadata !8, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32)), !dbg !12
  ; CHECK: DBG_VALUE debug-use %r0, debug-use _, !6, !DIExpression(DW_OP_LLVM_fragment, 0, 32)
  %r.sroa.0.0.insert.ext35 = zext i32 %extract.t84 to i64
  ret i64 %r.sroa.0.0.insert.ext35
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "LLVM", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2)
!1 = !DIFile(filename: "f.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 29, type: !7, isLocal: false, isDefinition: true, scopeLine: 30, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocalVariable(name: "r", scope: !5, file: !1, line: 37, type: !11)
!11 = distinct !DICompositeType(tag: DW_TAG_union_type, file: !1, line: 33, size: 64, elements: !2)
!12 = !DILocation(line: 37, column: 12, scope: !5)
