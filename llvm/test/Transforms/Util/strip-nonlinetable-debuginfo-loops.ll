; RUN: opt -S -strip-nonlinetable-debuginfo %s -o %t
; RUN: cat %t | FileCheck %s
; RUN: cat %t | FileCheck %s --check-prefix=NEGATIVE
; void f(volatile int *i) {
;   while (--*i) {}
; }
source_filename = "/tmp/loop.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

define void @f(i32* %i) local_unnamed_addr #0 !dbg !7 {
entry:
  tail call void @llvm.dbg.value(metadata i32* %i, i64 0, metadata !14, metadata !15), !dbg !16
  br label %while.cond, !dbg !17

while.cond:                                       ; preds = %while.cond, %entry
  %0 = load volatile i32, i32* %i, align 4, !dbg !18, !tbaa !19
  %dec = add nsw i32 %0, -1, !dbg !18
  store volatile i32 %dec, i32* %i, align 4, !dbg !18, !tbaa !19
  %tobool = icmp eq i32 %dec, 0, !dbg !17
  ; CHECK: !llvm.loop ![[LOOP:[0-9]+]]
  br i1 %tobool, label %while.end, label %while.cond, !dbg !17, !llvm.loop !23

while.end:                                        ; preds = %while.cond
  ret void, !dbg !25
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

; CHECK: ![[CU:.*]] = distinct !DICompileUnit(language: DW_LANG_C99,
; CHECK-SAME:                                 emissionKind: LineTablesOnly
; NEGATIVE-NOT: !DICompileUnit({{.*}} emissionKind: FullDebug
!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 5.0.0 (trunk 298880) (llvm/trunk 298875)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "/tmp/loop.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"clang version 5.0.0 (trunk 298880) (llvm/trunk 298875)"}
; CHECK: ![[F:[0-9]]] = distinct !DISubprogram(name: "f", scope: !1
!7 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !13)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!11 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !12)
; NEGATIVE-NOT: !DIBasicType(name: "int",
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{!14}
!14 = !DILocalVariable(name: "i", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!15 = !DIExpression()
!16 = !DILocation(line: 1, column: 22, scope: !7)
; CHECK: ![[BEGIN:[0-9]+]] = !DILocation(line: 2, column: 3, scope: ![[F]])
!17 = !DILocation(line: 2, column: 3, scope: !7)
!18 = !DILocation(line: 2, column: 10, scope: !7)
!19 = !{!20, !20, i64 0}
!20 = !{!"int", !21, i64 0}
!21 = !{!"omnipotent char", !22, i64 0}
!22 = !{!"Simple C/C++ TBAA"}
; CHECK: ![[LOOP]] = distinct !{![[LOOP]], ![[BEGIN]], ![[END:[0-9]+]]}
!23 = distinct !{!23, !17, !24}
; CHECK: ![[END]] = !DILocation(line: 3, column: 3, scope: ![[F]])
!24 = !DILocation(line: 3, column: 3, scope: !7)
!25 = !DILocation(line: 4, column: 1, scope: !7)
