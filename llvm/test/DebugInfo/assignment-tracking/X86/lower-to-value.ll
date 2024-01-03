; RUN: llc %s -stop-before finalize-isel -o - \
; RUN:    -experimental-debug-variable-locations=false \
; RUN:    -debug-ata-coalesce-frags=true \
; RUN: | FileCheck %s --check-prefixes=CHECK,DBGVALUE --implicit-check-not=DBG_VALUE

; RUN: llc --try-experimental-debuginfo-iterators %s -stop-before finalize-isel -o - \
; RUN:    -experimental-debug-variable-locations=false \
; RUN:    -debug-ata-coalesce-frags=true \
; RUN: | FileCheck %s --check-prefixes=CHECK,DBGVALUE --implicit-check-not=DBG_VALUE
; RUN: llc %s -stop-before finalize-isel -o - \
; RUN:    -experimental-debug-variable-locations=true \
; RUN: | FileCheck %s --check-prefixes=CHECK,INSTRREF --implicit-check-not=DBG_VALUE \
; RUN:    --implicit-check-not=DBG_INSTR_REF


; RUN: llc --try-experimental-debuginfo-iterators %s -stop-before finalize-isel -o - \
; RUN:    -experimental-debug-variable-locations=true \
; RUN: | FileCheck %s --check-prefixes=CHECK,INSTRREF --implicit-check-not=DBG_VALUE \
; RUN:    --implicit-check-not=DBG_INSTR_REF

;; Check that dbg.assigns for an aggregate variable which lives on the stack
;; for some of its lifetime are lowered into an appropriate set of DBG_VALUEs.
;;
;; $ cat test.cpp
;; void esc(long* p);
;; struct Ex {
;;   long A;
;;   long B;
;; };
;; long fun() {
;;   Ex X;
;;   X.B = 0;
;;   esc(&X.B);
;;   X.B += 2;
;;   return X.B;
;; }
;; $ clang++ test.cpp -O2 -g -emit-llvm -S -c -Xclang -fexperimental-assignment-tracking

; CHECK: ![[VAR:[0-9]+]] = !DILocalVariable(name: "X",
;; Check we have no debug info for local in the side table.
; CHECK: stack:
; CHECK-NEXT: - { id: 0, name: X, type: default, offset: 0, size: 16, alignment: 8,
; CHECK-NEXT:     stack-id: default, callee-saved-register: '', callee-saved-restored: true,
; CHECK-NEXT:     debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }

;; Initially the whole variable is on the stack.
; CHECK: bb.0.entry:
; CHECK-NEXT: DBG_VALUE %stack.0.X, $noreg, ![[VAR]], !DIExpression(DW_OP_deref), debug-location

;; Then there is a store to the upper 64 bits.
; CHECK: MOV64mi32 %stack.0.X, 1, $noreg, 8, $noreg, 0, debug-location
;; No change in variable location: the stack home is still valid.

;; The final assignment (X.B += 2) doesn't get stored back to the alloca. This
;; means that that the stack location isn't valid for the entire lifetime of X.
; DBGVALUE: %2:gr64 = nsw ADD64ri32 %1, 2, implicit-def dead $eflags, debug-location
; DBGVALUE-NEXT: DBG_VALUE %2, $noreg, ![[VAR]], !DIExpression(DW_OP_LLVM_fragment, 64, 64), debug-location
; INSTRREF: %2:gr64 = nsw ADD64ri32 %1, 2, implicit-def dead $eflags, debug-instr-number 1
; INSTRREF-NEXT: DBG_INSTR_REF ![[VAR]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_fragment, 64, 64), dbg-instr-ref(1, 0), debug-location

;; Bits [0, 64) are still stack homed. FIXME, this particular reinstatement is
;; unnecessary.
; CHECK-NEXT: DBG_VALUE %stack.0.X, $noreg, ![[VAR]], !DIExpression(DW_OP_deref, DW_OP_LLVM_fragment, 0, 64)

source_filename = "test.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Ex = type { i64, i64 }

define dso_local i64 @_Z3funv() local_unnamed_addr !dbg !7 {
entry:
  %X = alloca %struct.Ex, align 8, !DIAssignID !17
  call void @llvm.dbg.assign(metadata i1 undef, metadata !12, metadata !DIExpression(), metadata !17, metadata ptr %X, metadata !DIExpression()), !dbg !18
  %B = getelementptr inbounds %struct.Ex, ptr %X, i64 0, i32 1, !dbg !20
  store i64 0, ptr %B, align 8, !dbg !21, !DIAssignID !27
  call void @llvm.dbg.assign(metadata i64 0, metadata !12, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 64), metadata !27, metadata ptr %B, metadata !DIExpression()), !dbg !21
  call void @_Z3escPl(ptr nonnull %B), !dbg !28
  %0 = load i64, ptr %B, align 8, !dbg !29
  %add = add nsw i64 %0, 2, !dbg !29
  call void @llvm.dbg.assign(metadata i64 %add, metadata !12, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 64), metadata !30, metadata ptr %B, metadata !DIExpression()), !dbg !29
  ret i64 %add, !dbg !32
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, ptr nocapture)
declare !dbg !33 dso_local void @_Z3escPl(ptr) local_unnamed_addr
declare void @llvm.lifetime.end.p0i8(i64 immarg, ptr nocapture)
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !1000}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 12.0.0"}
!7 = distinct !DISubprogram(name: "fun", linkageName: "_Z3funv", scope: !1, file: !1, line: 6, type: !8, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DILocalVariable(name: "X", scope: !7, file: !1, line: 7, type: !13)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Ex", file: !1, line: 2, size: 128, flags: DIFlagTypePassByValue, elements: !14, identifier: "_ZTS2Ex")
!14 = !{!15, !16}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "A", scope: !13, file: !1, line: 3, baseType: !10, size: 64)
!16 = !DIDerivedType(tag: DW_TAG_member, name: "B", scope: !13, file: !1, line: 4, baseType: !10, size: 64, offset: 64)
!17 = distinct !DIAssignID()
!18 = !DILocation(line: 0, scope: !7)
!19 = !DILocation(line: 7, column: 3, scope: !7)
!20 = !DILocation(line: 8, column: 5, scope: !7)
!21 = !DILocation(line: 8, column: 7, scope: !7)
!27 = distinct !DIAssignID()
!28 = !DILocation(line: 9, column: 3, scope: !7)
!29 = !DILocation(line: 10, column: 7, scope: !7)
!30 = distinct !DIAssignID()
!31 = !DILocation(line: 12, column: 1, scope: !7)
!32 = !DILocation(line: 11, column: 3, scope: !7)
!33 = !DISubprogram(name: "esc", linkageName: "_Z3escPl", scope: !1, file: !1, line: 1, type: !34, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!34 = !DISubroutineType(types: !35)
!35 = !{null, !36}
!36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
