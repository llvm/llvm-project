; RUN: llc %s -o - -stop-after=finalize-isel \
; RUN: | FileCheck %s --implicit-check-not=DBG

;; In the IR below, for variable n, we get dbg intrinsics that describe this:
;;
;; entry-block:
;;     Frag (off=0,  sz=32): non-undef
;;     Frag (off=64, sz=64): undef
;;     Frag (off=64, sz=32): non-undef
;;
;; The undef is redundant, as it doesn't close any open location ranges. Check
;; that it has been removed. Removing redundant undefs from the entry block
;; helps avoid losing coverage due to SelectionDAG doing weird (/bad) things.
;; Even if SelectionDAG is fixed, fewer redundant DBG instructions is still a
;; valuable goal.

;; The test
;; --------
;; We expect to see two DBG instructions, one for each non-undef fragment. We
;; don't bother checking the operands because it doesn't matter if either of
;; these have become undef as a result of SelectionDAG dropping the values
;; (which happens to be the case here). It's just important that SelectionDAG
;; was fed these fragments.

; CHECK: DBG{{.*}}DIExpression({{(DW_OP_LLVM_arg, 0, )?}}DW_OP_LLVM_fragment, 0, 32)
; CHECK: DBG{{.*}}DIExpression({{(DW_OP_LLVM_arg, 0, )?}}DW_OP_LLVM_fragment, 64, 32)

;; Source
;; ------
;; IR llvm-reduced from optimized IR generated from, itself reduced from
;; CTMark's bullet source file btScaledBvhTriangleMeshShape.cpp:
;; class a {
;; public:
;;   float b[4];
;;   __attribute__((nodebug)) a() {}
;;   __attribute__((nodebug)) a(float c, float p2) {
;;     b[0] = c;
;;     b[2] = p2;
;;   }
;;   __attribute__((nodebug)) void operator+=(a) {
;;     b[0] += 0;
;;     b[2] += 2;
;;   }
;;   __attribute__((nodebug)) float d(a c) { return c.b[0] + c.b[2]; }
;; };
;;
;; __attribute__((nodebug)) void operator-(a, a);
;; __attribute__((nodebug)) a operator*(float, a p2) {
;;   a e(p2.b[0], p2.b[2]);
;;   return e;
;; }
;;
;; __attribute__((nodebug)) a x();
;; __attribute__((nodebug)) a y(int);
;;
;; void k() {
;;   __attribute__((nodebug)) a l = x();
;;   __attribute__((nodebug)) a m = l;
;;   __attribute__((nodebug)) a ag;
;;   a n = 0.f * m;
;;
;;   n += a();
;;   __attribute__((nodebug)) a ah(y(0).d(n), 0);
;;   ag - ah;
;; }
target triple = "x86_64-unknown-linux-gnu"

define void @_Z1kv({ <2 x float>, <2 x float> } %call, <2 x float> %0, float %n.sroa.6.8.vec.extract) !dbg !7 {
entry:
  %call1 = tail call { <2 x float>, <2 x float> } poison(), !dbg !13
  %1 = extractvalue { <2 x float>, <2 x float> } %call, 1
  %add.i = fadd float poison, 0.000000e+00
  call void @llvm.dbg.assign(metadata float %add.i, metadata !11, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32), metadata !14, metadata ptr undef, metadata !DIExpression()), !dbg !15
  %n.sroa.6.8.vec.extract2 = extractelement <2 x float> %0, i64 0
  %add4.i = fadd float %n.sroa.6.8.vec.extract, 0.000000e+00
  call void @llvm.dbg.value(metadata <2 x float> undef, metadata !11, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 64)), !dbg !15
  call void @llvm.dbg.assign(metadata float %add4.i, metadata !11, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 32), metadata !16, metadata ptr undef, metadata !DIExpression()), !dbg !15
  %add.i23 = fadd float 0.000000e+00, 0.000000e+00
  %ah.sroa.0.0.vec.insert = insertelement <2 x float> zeroinitializer, float %add4.i, i64 0
  tail call void poison(<2 x float> zeroinitializer, <2 x float> zeroinitializer, <2 x float> %ah.sroa.0.0.vec.insert, <2 x float> zeroinitializer)
  ret void
}

declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !1000}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "reduce.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 1}
!6 = !{i32 7, !"frame-pointer", i32 2}
!7 = distinct !DISubprogram(name: "k", linkageName: "_Z1kv", scope: !1, file: !1, line: 25, type: !8, scopeLine: 25, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{!11}
!11 = !DILocalVariable(name: "n", scope: !7, file: !1, line: 29, type: !12)
!12 = !DICompositeType(tag: DW_TAG_class_type, name: "a", file: !1, line: 1, size: 128, flags: DIFlagFwdDecl | DIFlagNonTrivial, identifier: "_ZTS1a")
!13 = !DILocation(line: 26, scope: !7)
!14 = distinct !DIAssignID()
!15 = !DILocation(line: 0, scope: !7)
!16 = distinct !DIAssignID()
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
