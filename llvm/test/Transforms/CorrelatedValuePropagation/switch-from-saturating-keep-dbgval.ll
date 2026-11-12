; RUN: opt < %s -passes=correlated-propagation -S | FileCheck %s

; When optimisations change saturating arithmetic to normal arithmetic, debug values
; need to be updated to prevent incorrect values in the debugger
; See issue #220562

define dso_local i32 @saturating_inst(i32 noundef %x) #0 !dbg !9 {
; CHECK:      entry:
; CHECK-NEXT: %elt.sat1 = sub nuw i32 %x, 10
; CHECK-NEXT: #dbg_value(i32 poison, [[META1:![0-9]+]], !DIExpression(), [[META2:![0-9]+]])
entry:
  %elt.sat = call i32 @llvm.usub.sat.i32(i32 %x, i32 10)
    #dbg_value(i32 %elt.sat, !21, !DIExpression(), !19)
  %cmp = icmp ult i32 %x, 10
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %result.0 = phi i32 [ 42, %if.then ], [ %elt.sat, %entry ]
  ret i32 %result.0
}

declare i32 @llvm.usub.sat.i32(i32, i32) #1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 24.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "case.c", directory: "/")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!9 = distinct !DISubprogram(name: "saturating_inst", scope: !1, file: !1, line: 3, type: !10, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !17)
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !12}
!12 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint32_t", file: !13, line: 26, baseType: !14)
!13 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/stdint-uintn.h", directory: "")
!14 = !DIDerivedType(tag: DW_TAG_typedef, name: "__uint32_t", file: !15, line: 42, baseType: !16)
!15 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types.h", directory: "")
!16 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!17 = !{}
!19 = !DILocation(line: 0, scope: !9)
!21 = !DILocalVariable(name: "sub", scope: !9, file: !1, line: 4, type: !12)
; CHECK: [[META3:![0-9]+]] = distinct !DISubprogram(name: "saturating_inst"
; CHECK: [[META1]] = !DILocalVariable(name: "sub", scope: [[META3]]
; CHECK: [[META2]] = !DILocation(line: 0, scope: [[META3]])
