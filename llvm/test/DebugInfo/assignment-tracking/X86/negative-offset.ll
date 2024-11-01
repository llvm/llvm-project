; RUN: llc %s -stop-after=finalize-isel -o - | FileCheck %s --implicit-check-not=DBG_VALUE

;; Check stores to an address computed as a negative offset from an alloca are
;; ignored by the assignment tracking analysis. For this example that should
;; result in no DBG_VALUEs in the while.body.lr.ph branch.
;;
;; See llvm.org/PR62838 for more info.
;;
;; $ clang test.c -O1 -g
;; $ cat tes.c
;; void a(int *p);
;; __attribute__((nodebug)) int b;
;;
;; int main() {
;;   int c[1];
;;   __attribute__((nodebug)) int d = -1;
;;
;;   while (b) {
;;     c[0] = 0;
;;     c[d] = 0;
;;   }
;;   a(c);
;;   return 0;
;; }

; CHECK: bb.0.entry:
; CHECK-NEXT: successors
; CHECK-NEXT: {{^ *$}}
; CHECK-NEXT: DBG_VALUE %stack.0.c, $noreg, ![[#]], !DIExpression(DW_OP_deref)

; CHECK:  bb.2.while.body:
; CHECK-NEXT: successors
; CHECK-NEXT: {{^ *$}}
; CHECK-NEXT: DBG_VALUE 0, $noreg, ![[#]], !DIExpression()

target triple = "x86_64-unknown-linux-gnu"

@b = dso_local local_unnamed_addr global i32 0, align 4

define dso_local i32 @main() local_unnamed_addr #0 !dbg !10 {
entry:
  %c = alloca [1 x i32], align 4, !DIAssignID !19
  call void @llvm.dbg.assign(metadata i1 undef, metadata !15, metadata !DIExpression(), metadata !19, metadata ptr %c, metadata !DIExpression()), !dbg !20
  %0 = load i32, ptr @b, align 4
  %tobool.not = icmp eq i32 %0, 0
  br i1 %tobool.not, label %while.end, label %while.body.lr.ph

while.body.lr.ph:                                 ; preds = %entry
  %arrayidx1 = getelementptr inbounds [1 x i32], ptr %c, i64 0, i64 -1
  store i32 0, ptr %arrayidx1, align 4
  br label %while.body

while.body:                                       ; preds = %while.body, %while.body.lr.ph
  call void @llvm.dbg.assign(metadata i32 0, metadata !15, metadata !DIExpression(), metadata !28, metadata ptr %c, metadata !DIExpression()), !dbg !20
  br label %while.body

while.end:                                        ; preds = %entry
  call void @a(ptr noundef nonnull %c)
  ret i32 0
}

declare !dbg !31 void @a(ptr noundef) local_unnamed_addr #2
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata) #3

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 17.0.0)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!9 = !{!"clang version 17.0.0"}
!10 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 4, type: !11, scopeLine: 4, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14)
!11 = !DISubroutineType(types: !12)
!12 = !{!13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{!15}
!15 = !DILocalVariable(name: "c", scope: !10, file: !1, line: 5, type: !16)
!16 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 32, elements: !17)
!17 = !{!18}
!18 = !DISubrange(count: 1)
!19 = distinct !DIAssignID()
!20 = !DILocation(line: 0, scope: !10)
!28 = distinct !DIAssignID()
!31 = !DISubprogram(name: "a", scope: !1, file: !1, line: 1, type: !32, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !35)
!32 = !DISubroutineType(types: !33)
!33 = !{null, !34}
!34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!35 = !{}
