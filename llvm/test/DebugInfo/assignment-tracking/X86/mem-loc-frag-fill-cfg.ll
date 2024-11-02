; RUN: llc %s -stop-before finalize-isel -o - \
; RUN:    -experimental-assignment-tracking \
; RUN:    -experimental-debug-variable-locations=false \
; RUN: | FileCheck %s --implicit-check-not=DBG_
; RUN: llc %s -stop-before finalize-isel -o - \
; RUN:    -experimental-assignment-tracking \
; RUN:    -experimental-debug-variable-locations=true \
; RUN: | FileCheck %s --implicit-check-not=DBG_

;; Check that the mem-loc-frag-fill pseudo-pass works on a simple CFG. When
;; LLVM sees a dbg.value with an overlapping fragment it essentially considers
;; the previous location as valid for all bits in that fragment. The pass
;; inserts dbg.value fragments to preserve memory locations for bits in memory
;; when overlapping fragments are encountered.

;; nums lives in mem, except prior to the second call to step() where there has
;; been some DSE. At this point, the memory loc for nums.c is invalid.  But the
;; rest of num's bits, [0, 64), are in memory, so check there's a dbg.value for
;; them.

;; $ cat test.cpp
;; struct Nums { int a, b, c; };
;;
;; void esc1(struct Nums*);
;; void esc2(struct Nums*);
;; bool step();
;;
;; int main() {
;;   struct Nums nums = { 1, 2, 1 };
;;   if (step())
;;     esc1(&nums);
;;   else
;;     esc2(&nums);
;;
;;   nums.c = 2; //< Include some DSE to force a non-mem location.
;;   step();
;;
;;   nums.c = nums.a;
;;
;;   esc1(&nums);
;;   return 0;
;; }
;;
;; $ clang++ test.cpp -O2 -g -Xclang -fexperimental-assignment-tracking -emit-llvm -S -o -

;; Most check lines are inline in main.
; CHECK: ![[nums:[0-9]+]] = !DILocalVariable(name: "nums",

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Nums = type { i32, i32, i32 }

@__const.main.nums = private unnamed_addr constant %struct.Nums { i32 1, i32 2, i32 1 }, align 4

declare void @_Z4esc1P4Nums(ptr nocapture noundef readonly %p)
declare void @_Z4esc2P4Nums(ptr nocapture noundef readonly %p)

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #3 !dbg !40 {
; CHECK: name: main
entry:
  %nums = alloca %struct.Nums, align 4, !DIAssignID !45
  call void @llvm.dbg.assign(metadata i1 undef, metadata !44, metadata !DIExpression(), metadata !45, metadata ptr %nums, metadata !DIExpression()), !dbg !46
; CHECK: DBG_VALUE %stack.0.nums, $noreg, ![[nums]], !DIExpression(DW_OP_deref)
  call void @llvm.memcpy.p0i8.p0i8.i64(ptr noundef nonnull align 4 dereferenceable(12) %nums, ptr noundef nonnull align 4 dereferenceable(12) %nums, i64 12, i1 false), !dbg !48, !DIAssignID !49
  call void @llvm.dbg.assign(metadata i1 undef, metadata !44, metadata !DIExpression(), metadata !49, metadata ptr %nums, metadata !DIExpression()), !dbg !46
  %call = tail call noundef zeroext i1 @_Z4stepv(), !dbg !50
  br i1 %call, label %if.then, label %if.else, !dbg !52

if.then:                                          ; preds = %entry
  call void @_Z4esc1P4Nums(ptr noundef nonnull %nums), !dbg !53
  br label %if.end, !dbg !53

if.else:                                          ; preds = %entry
  call void @_Z4esc2P4Nums(ptr noundef nonnull %nums), !dbg !54
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
; CHECK: bb.3.if.end:
; CHECK-NEXT: DBG_VALUE 2, $noreg, ![[nums]], !DIExpression(DW_OP_LLVM_fragment, 64, 32), debug-location
; CHECK-NEXT: DBG_VALUE %stack.0.nums, $noreg, ![[nums]], !DIExpression(DW_OP_deref, DW_OP_LLVM_fragment, 0, 64)
  %c = getelementptr inbounds %struct.Nums, ptr %nums, i64 0, i32 2, !dbg !55
  call void @llvm.dbg.assign(metadata i32 2, metadata !44, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 32), metadata !56, metadata ptr %c, metadata !DIExpression()), !dbg !46
  %call1 = tail call noundef zeroext i1 @_Z4stepv(), !dbg !57
  store i32 1, ptr %c, align 4, !dbg !58, !DIAssignID !61
; CHECK:      MOV32mi %stack.0.nums, 1, $noreg, 8, $noreg, 1
; CHECK-NEXT: DBG_VALUE %stack.0.nums, $noreg, ![[nums]], !DIExpression(DW_OP_plus_uconst, 8, DW_OP_deref, DW_OP_LLVM_fragment, 64, 32)
  call void @llvm.dbg.assign(metadata i32 1, metadata !44, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 32), metadata !61, metadata ptr %c, metadata !DIExpression()), !dbg !46
  call void @_Z4esc1P4Nums(ptr noundef nonnull %nums), !dbg !62
  ret i32 0, !dbg !64
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, ptr nocapture) #4
declare !dbg !65 dso_local noundef zeroext i1 @_Z4stepv() local_unnamed_addr #5
declare void @llvm.lifetime.end.p0i8(i64 immarg, ptr nocapture) #4
declare void @llvm.memcpy.p0i8.p0i8.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #1
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata) #2

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !12, !13, !14}
!llvm.ident = !{!15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "glob", scope: !2, file: !3, line: 2, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.cpp", directory: "/")
!4 = !{!0}
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Nums", file: !3, line: 1, size: 96, flags: DIFlagTypePassByValue, elements: !6, identifier: "_ZTS4Nums")
!6 = !{!7, !9, !10}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !5, file: !3, line: 1, baseType: !8, size: 32)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !5, file: !3, line: 1, baseType: !8, size: 32, offset: 32)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !5, file: !3, line: 1, baseType: !8, size: 32, offset: 64)
!11 = !{i32 7, !"Dwarf Version", i32 5}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{i32 7, !"uwtable", i32 1}
!15 = !{!"clang version 14.0.0"}
!40 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 7, type: !41, scopeLine: 7, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !43)
!41 = !DISubroutineType(types: !42)
!42 = !{!8}
!43 = !{!44}
!44 = !DILocalVariable(name: "nums", scope: !40, file: !3, line: 8, type: !5)
!45 = distinct !DIAssignID()
!46 = !DILocation(line: 0, scope: !40)
!47 = !DILocation(line: 8, column: 3, scope: !40)
!48 = !DILocation(line: 8, column: 15, scope: !40)
!49 = distinct !DIAssignID()
!50 = !DILocation(line: 9, column: 7, scope: !51)
!51 = distinct !DILexicalBlock(scope: !40, file: !3, line: 9, column: 7)
!52 = !DILocation(line: 9, column: 7, scope: !40)
!53 = !DILocation(line: 10, column: 5, scope: !51)
!54 = !DILocation(line: 12, column: 5, scope: !51)
!55 = !DILocation(line: 14, column: 8, scope: !40)
!56 = distinct !DIAssignID()
!57 = !DILocation(line: 15, column: 3, scope: !40)
!58 = !DILocation(line: 17, column: 10, scope: !40)
!61 = distinct !DIAssignID()
!62 = !DILocation(line: 19, column: 3, scope: !40)
!63 = !DILocation(line: 21, column: 1, scope: !40)
!64 = !DILocation(line: 20, column: 3, scope: !40)
!65 = !DISubprogram(name: "step", linkageName: "_Z4stepv", scope: !3, file: !3, line: 5, type: !66, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !69)
!66 = !DISubroutineType(types: !67)
!67 = !{!68}
!68 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!69 = !{}
