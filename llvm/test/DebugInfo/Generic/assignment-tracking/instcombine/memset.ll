; RUN: opt %s -S -passes=instcombine -o - \
; RUN: | FileCheck %s

;; $ cat test.cpp
;; void esc(int*);
;; void fun() {
;;   int local[5];
;;   __builtin_memset(local, 8, 4 * 4);
;;   __builtin_memset(local, 0, 2 * 4);
;;   esc(local);
;; }
;; IR grabbed before instcombine in:
;; clang++ -O2 -g -Xclang -fexperimental-assignment-tracking

;; Instcombine is going to turn the second memset into a store. Check that it
;; inherits the DIAssignID from the memset and that the dbg.assign's value
;; component is correct.

; CHECK:      store i64 0, ptr %local, align 16{{.*}}, !DIAssignID ![[ID:[0-9]+]]
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i64 0, metadata !{{.*}}, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 64), metadata ![[ID]], metadata ptr %local, metadata !DIExpression())

define dso_local void @_Z3funv() local_unnamed_addr #0 !dbg !7 {
entry:
  %local = alloca [5 x i32], align 16, !DIAssignID !16
  call void @llvm.dbg.assign(metadata i1 undef, metadata !11, metadata !DIExpression(), metadata !16, metadata ptr %local, metadata !DIExpression()), !dbg !17
  %0 = bitcast ptr %local to ptr, !dbg !18
  call void @llvm.lifetime.start.p0i8(i64 20, ptr %0) #5, !dbg !18
  %arraydecay = getelementptr inbounds [5 x i32], ptr %local, i64 0, i64 0, !dbg !19
  %1 = bitcast ptr %arraydecay to ptr, !dbg !19
  call void @llvm.memset.p0i8.i64(ptr align 16 %1, i8 8, i64 16, i1 false), !dbg !19, !DIAssignID !20
  call void @llvm.dbg.assign(metadata i8 8, metadata !11, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 128), metadata !20, metadata ptr %1, metadata !DIExpression()), !dbg !17
  call void @llvm.memset.p0i8.i64(ptr align 16 %1, i8 0, i64 8, i1 false), !dbg !21, !DIAssignID !22
  call void @llvm.dbg.assign(metadata i8 0, metadata !11, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 64), metadata !22, metadata ptr %1, metadata !DIExpression()), !dbg !17
  call void @_Z3escPi(ptr noundef %arraydecay), !dbg !23
  call void @llvm.lifetime.end.p0i8(i64 20, ptr %0) #5, !dbg !24
  ret void, !dbg !24
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, ptr nocapture)
declare void @llvm.memset.p0i8.i64(ptr nocapture writeonly, i8, i64, i1 immarg)
declare !dbg !25 dso_local void @_Z3escPi(ptr noundef) local_unnamed_addr
declare void @llvm.lifetime.end.p0i8(i64 immarg, ptr nocapture)
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !1000}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 1}
!6 = !{!"clang version 14.0.0"}
!7 = distinct !DISubprogram(name: "fun", linkageName: "_Z3funv", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{!11}
!11 = !DILocalVariable(name: "local", scope: !7, file: !1, line: 3, type: !12)
!12 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 160, elements: !14)
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{!15}
!15 = !DISubrange(count: 5)
!16 = distinct !DIAssignID()
!17 = !DILocation(line: 0, scope: !7)
!18 = !DILocation(line: 3, column: 3, scope: !7)
!19 = !DILocation(line: 4, column: 3, scope: !7)
!20 = distinct !DIAssignID()
!21 = !DILocation(line: 5, column: 3, scope: !7)
!22 = distinct !DIAssignID()
!23 = !DILocation(line: 6, column: 3, scope: !7)
!24 = !DILocation(line: 7, column: 1, scope: !7)
!25 = !DISubprogram(name: "esc", linkageName: "_Z3escPi", scope: !1, file: !1, line: 1, type: !26, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !29)
!26 = !DISubroutineType(types: !27)
!27 = !{null, !28}
!28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!29 = !{}
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
