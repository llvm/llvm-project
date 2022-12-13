; RUN: llc -stop-after=finalize-isel %s -o - \
; RUN:    -experimental-assignment-tracking \
; RUN: | FileCheck %s

;; Check that a dbg.assign for a fully stack-homed variable causes the variable
;; location to appear in the Machine Function side table.
;;
;; $ cat test.cpp
;; void maybe_writes(int*);
;; void ext(int, int, int, int, int, int, int, int, int, int);
;; int example() {
;;    int local;
;;    maybe_writes(&local);
;;    ext(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
;;    return local;
;; }
;; $ clang++ -O2 -g -emit-llvm -S -c -Xclang -fexperimental-assignment-tracking

; CHECK: ![[VAR:[0-9]+]] = !DILocalVariable(name: "local",
; CHECK: stack:
; CHECK-NEXT: - { id: 0, name: local, type: default, offset: 0, size: 4, alignment: 4, 
; CHECK-NEXT:     stack-id: default, callee-saved-register: '', callee-saved-restored: true, 
; CHECK-NEXT:     debug-info-variable: '![[VAR]]', debug-info-expression: '!DIExpression()', 
; CHECK-NEXT:     debug-info-location: '!{{.+}}' }

source_filename = "test.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local i32 @_Z7examplev() local_unnamed_addr !dbg !7 {
entry:
  %local = alloca i32, align 4, !DIAssignID !13
  call void @llvm.dbg.assign(metadata i1 undef, metadata !12, metadata !DIExpression(), metadata !13, metadata ptr %local, metadata !DIExpression()), !dbg !14
  %0 = bitcast ptr %local to ptr, !dbg !15
  call void @llvm.lifetime.start.p0i8(i64 4, ptr nonnull %0), !dbg !15
  call void @_Z12maybe_writesPi(ptr nonnull %local), !dbg !16
  call void @_Z3extiiiiiiiiii(i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9), !dbg !17
  %1 = load i32, ptr %local, align 4, !dbg !18
  call void @llvm.lifetime.end.p0i8(i64 4, ptr nonnull %0), !dbg !23
  ret i32 %1, !dbg !24
}

declare !dbg !25 dso_local void @_Z12maybe_writesPi(ptr) local_unnamed_addr
declare !dbg !29 dso_local void @_Z3extiiiiiiiiii(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) local_unnamed_addr
declare void @llvm.lifetime.start.p0i8(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0i8(i64 immarg, ptr nocapture)
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 12.0.0"}
!7 = distinct !DISubprogram(name: "example", linkageName: "_Z7examplev", scope: !1, file: !1, line: 3, type: !8, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DILocalVariable(name: "local", scope: !7, file: !1, line: 4, type: !10)
!13 = distinct !DIAssignID()
!14 = !DILocation(line: 0, scope: !7)
!15 = !DILocation(line: 4, column: 4, scope: !7)
!16 = !DILocation(line: 5, column: 4, scope: !7)
!17 = !DILocation(line: 6, column: 4, scope: !7)
!18 = !DILocation(line: 7, column: 11, scope: !7)
!23 = !DILocation(line: 8, column: 1, scope: !7)
!24 = !DILocation(line: 7, column: 4, scope: !7)
!25 = !DISubprogram(name: "maybe_writes", linkageName: "_Z12maybe_writesPi", scope: !1, file: !1, line: 1, type: !26, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!26 = !DISubroutineType(types: !27)
!27 = !{null, !28}
!28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!29 = !DISubprogram(name: "ext", linkageName: "_Z3extiiiiiiiiii", scope: !1, file: !1, line: 2, type: !30, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!30 = !DISubroutineType(types: !31)
!31 = !{null, !10, !10, !10, !10, !10, !10, !10, !10, !10, !10}
