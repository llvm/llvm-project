; RUN: llc %s -stop-after=finalize-isel -o - -experimental-assignment-tracking \
; RUN: | FileCheck %s

;; Hand written test because the scenario is unlikely. Check that the "value"
;; of a debug def PHIs is "undef" (because we don't actually track PHIs).
;;
;; entry:
;;   memdef 5, !1
;;   dbgdef !1
;;   br if.then, exit
;;
;; if.then:
;;   memdef 0, !1
;;   dbgdef !1
;;   br exit
;;
;; exit:
;;                  ; <-- Dbg=!1, Stack=!1, Loc=Mem
;;   memddef 1, !2  ; <-- Dbg=!1, Stack=!2, Loc=Val(undef) @HERE
;;   call
;;   dbgdef !2
;;
;; Check that the dbg.value inserted at @HERE is undef because there's no
;; appropriate alternative value to choose.

; CHECK: bb.0.entry:
; CHECK: DBG_VALUE %stack.0.c, $noreg, ![[var:[0-9]+]], !DIExpression(DW_OP_deref), debug-location
; CHECK-NEXT: MOV8mi %stack.0.c, 1, $noreg, 0, $noreg, 5, debug-location
; CHECL-NEXT: DBG_VALUE %stack.0.c, $noreg, ![[var]], !DIExpression(DW_OP_deref), debug-location

; CHECK: bb.1.if.then:
; CHECK: MOV8mi %stack.0.c, 1, $noreg, 0, $noreg, 0
; CHECK: DBG_VALUE %stack.0.c, $noreg, ![[var]], !DIExpression(DW_OP_deref), debug-location

; CHECK:  bb.2.exit:
; CHECK-NEXT: MOV8mi %stack.0.c, 1, $noreg, 0, $noreg, 1
;; @HERE
; CHECK-NEXT: DBG_VALUE $noreg, $noreg, ![[var]], !DIExpression()
; CHECK: CALL64pcrel32 @d
; CHECK-NEXT: ADJCALLSTACKUP64
; CHECK-NEXT: DBG_VALUE %stack.0.c, $noreg, ![[var]], !DIExpression(DW_OP_deref), debug-location

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define dso_local void @b(i1 %cond) local_unnamed_addr #0 !dbg !7 {
entry:
  %c = alloca i8, align 1, !DIAssignID !13
  call void @llvm.dbg.assign(metadata i1 undef, metadata !11, metadata !DIExpression(), metadata !13, metadata ptr %c, metadata !DIExpression()), !dbg !14
  store i8 5, ptr %c, align 1, !dbg !16, !DIAssignID !31
  call void @llvm.dbg.assign(metadata i8 5, metadata !11, metadata !DIExpression(), metadata !31, metadata ptr %c, metadata !DIExpression()), !dbg !14
  br i1 %cond, label %if.then, label %exit

if.then:
  tail call void (...) @d() #4, !dbg !21
  store i8 0, ptr %c, align 1, !dbg !16, !DIAssignID !31
  call void @llvm.dbg.assign(metadata i8 0, metadata !11, metadata !DIExpression(), metadata !31, metadata ptr %c, metadata !DIExpression()), !dbg !14
  br label %exit

exit:
  store i8 1, ptr %c, align 1, !dbg !16, !DIAssignID !20
  tail call void (...) @d() #4, !dbg !21
  call void @llvm.dbg.assign(metadata i8 1, metadata !11, metadata !DIExpression(), metadata !20, metadata ptr %c, metadata !DIExpression()), !dbg !14
  call void @a(ptr nonnull %c) #4, !dbg !22
  ret void, !dbg !23
}


declare !dbg !24 dso_local void @d(...) local_unnamed_addr #2
declare !dbg !27 dso_local void @a(ptr) local_unnamed_addr #2
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata) #3

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "reduce.c", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 12.0.0"}
!7 = distinct !DISubprogram(name: "b", scope: !1, file: !1, line: 3, type: !8, scopeLine: 3, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{!11}
!11 = !DILocalVariable(name: "c", scope: !7, file: !1, line: 4, type: !12)
!12 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!13 = distinct !DIAssignID()
!14 = !DILocation(line: 0, scope: !7)
!15 = !DILocation(line: 4, column: 3, scope: !7)
!16 = !DILocation(line: 4, column: 8, scope: !7)
!20 = distinct !DIAssignID()
!21 = !DILocation(line: 5, column: 3, scope: !7)
!22 = !DILocation(line: 6, column: 3, scope: !7)
!23 = !DILocation(line: 7, column: 1, scope: !7)
!24 = !DISubprogram(name: "d", scope: !1, file: !1, line: 2, type: !25, spFlags: DISPFlagOptimized, retainedNodes: !2)
!25 = !DISubroutineType(types: !26)
!26 = !{null, null}
!27 = !DISubprogram(name: "a", scope: !1, file: !1, line: 1, type: !28, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!28 = !DISubroutineType(types: !29)
!29 = !{null, !30}
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!31 = distinct !DIAssignID()
