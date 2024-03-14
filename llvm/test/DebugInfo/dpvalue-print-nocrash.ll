;; Tests that we can debug-print DPValues that have no markers attached.
; RUN: opt -passes="instcombine" -debug %s -o /dev/null 2>&1 | FileCheck %s
; REQUIRES: asserts

; CHECK: CLONE:   #dbg_value(

define ptr @func_10(i32 %p_11) {
entry:
  %conv108 = zext i32 %p_11 to i64
  tail call void @llvm.dbg.value(metadata i64 %conv108, metadata !4, metadata !DIExpression()), !dbg !12
  br label %func_29.exit

func_29.exit:                                     ; preds = %entry
  store i64 %conv108, ptr null, align 1
  ret ptr null
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2, globals: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "csmith5961503756960.c", directory: "/llvm")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DILocalVariable(name: "p_31", arg: 2, scope: !5, file: !1, line: 148, type: !7)
!5 = distinct !DISubprogram(name: "func_29", scope: !1, file: !1, line: 148, type: !6, scopeLine: 149, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!6 = !DISubroutineType(types: !2)
!7 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint64_t", file: !8, line: 60, baseType: !9)
!8 = !DIFile(filename: "/foo/_stdint.h", directory: "")
!9 = !DIDerivedType(tag: DW_TAG_typedef, name: "__uint64_t", file: !10, line: 108, baseType: !11)
!10 = !DIFile(filename: "/foo/_default_types.h", directory: "")
!11 = !DIBasicType(name: "unsigned long long", size: 64, encoding: DW_ATE_unsigned)
!12 = !DILocation(line: 0, scope: !5)
