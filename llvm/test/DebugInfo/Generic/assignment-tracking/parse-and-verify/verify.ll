; RUN: opt %s -S -passes=verify 2>&1 \
; RUN: | FileCheck %s

;; Check that badly formed assignment tracking metadata is caught either
;; while parsing or by the verifier.
;;
;; Checks for this one are inline.

define dso_local void @fun2() !dbg !15 {
  ;; DIAssignID copied here from @fun() where it is used by intrinsics.
  ; CHECK: dbg.assign not in same function as inst
  %x = alloca i32, align 4, !DIAssignID !14
  ret void
}

define dso_local void @fun() !dbg !7 {
entry:
  %a = alloca i32, align 4, !DIAssignID !14
  ;; Here something other than a dbg.assign intrinsic is using a DIAssignID.
  ; CHECK: !DIAssignID should only be used by llvm.dbg.assign intrinsics
  call void @llvm.dbg.value(metadata !14, metadata !10, metadata !DIExpression()), !dbg !13

  ;; Each following dbg.assign has an argument of the incorrect type.
  ; CHECK: invalid llvm.dbg.assign intrinsic address/value
  call void @llvm.dbg.assign(metadata !3, metadata !10, metadata !DIExpression(), metadata !14, metadata ptr undef, metadata !DIExpression()), !dbg !13
  ; CHECK: invalid llvm.dbg.assign intrinsic variable
  call void @llvm.dbg.assign(metadata i32 0, metadata !2, metadata !DIExpression(), metadata !14, metadata ptr undef, metadata !DIExpression()), !dbg !13
  ; CHECK: invalid llvm.dbg.assign intrinsic expression
  call void @llvm.dbg.assign(metadata !14, metadata !10, metadata !2, metadata !14, metadata ptr undef, metadata !DIExpression()), !dbg !13
  ; CHECK: invalid llvm.dbg.assign intrinsic DIAssignID
  call void @llvm.dbg.assign(metadata !14, metadata !10, metadata !DIExpression(), metadata !2, metadata ptr undef, metadata !DIExpression()), !dbg !13
  ; CHECK: invalid llvm.dbg.assign intrinsic address
  call void @llvm.dbg.assign(metadata !14, metadata !10, metadata !DIExpression(), metadata !14, metadata !3, metadata !DIExpression()), !dbg !13
  ;; Empty metadata debug operands are allowed.
  ; CHECK-NOT: invalid llvm.dbg.assign
  call void @llvm.dbg.assign(metadata !14, metadata !10, metadata !DIExpression(), metadata !14, metadata !2, metadata !DIExpression()), !dbg !13
  ; CHECK: invalid llvm.dbg.assign intrinsic address expression
  call void @llvm.dbg.assign(metadata !14, metadata !10, metadata !DIExpression(), metadata !14, metadata ptr undef, metadata !2), !dbg !13
  ret void
}

declare void @llvm.dbg.value(metadata, metadata, metadata)
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !1000}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 14.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 14.0.0"}
!7 = distinct !DISubprogram(name: "fun", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocalVariable(name: "local", scope: !7, file: !1, line: 2, type: !11)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocation(line: 1, column: 1, scope: !7)
!14 = distinct !DIAssignID()
!15 = distinct !DISubprogram(name: "fun2", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
