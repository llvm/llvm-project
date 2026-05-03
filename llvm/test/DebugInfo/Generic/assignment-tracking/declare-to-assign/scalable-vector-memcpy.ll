; RUN: opt -passes=declare-to-assign %s -S | FileCheck %s

;; Check that declare-to-assign does not crash when a fixed-size memcpy writes
;; into a scalable-vector alloca. getAssignmentInfoImpl must bail out when the
;; destination alloca's type is scalable, not only when the store size itself is
;; scalable. The dbg.declare for the scalable alloca should be preserved as-is.
;;
;; Derived from: https://github.com/llvm/llvm-project/issues/192728
;;   #include <string.h>
;;   #include <riscv_vector.h>
;;   vint32m1_t get_i32x4(int* v) {
;;     vint32m1_t r;
;;     memcpy(&r, v, 16);
;;     return r;
;;   }
;; Compiled with: clang -target riscv64-unknown-linux-gnu -march=rv64gcv -O1 -g

;; The scalable-vector alloca must not be annotated with DIAssignID.
; CHECK:     = alloca <vscale x 2 x i32>, align 4
; CHECK-NOT: DIAssignID
; CHECK-SAME: {{$}}
;; Its dbg.declare must be preserved (not converted to dbg.assign).
; CHECK:     #dbg_declare(ptr %r,
; CHECK-NOT: #dbg_assign(ptr %r,

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-linux-gnu"

define dso_local <vscale x 2 x i32> @get_i32x4(ptr noundef %0) !dbg !16 {
  %2 = alloca ptr, align 8
  %r = alloca <vscale x 2 x i32>, align 4
  store ptr %0, ptr %2, align 8, !tbaa !31
    #dbg_declare(ptr %2, !29, !DIExpression(), !34)
  call void @llvm.lifetime.start.p0(ptr %r), !dbg !35
    #dbg_declare(ptr %r, !30, !DIExpression(), !36)
  %3 = load ptr, ptr %2, align 8, !dbg !37, !tbaa !31
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %r, ptr align 4 %3, i64 16, i1 false), !dbg !38
  %4 = load <vscale x 2 x i32>, ptr %r, align 4, !dbg !39, !tbaa !40
  call void @llvm.lifetime.end.p0(ptr %r), !dbg !42
  ret <vscale x 2 x i32> %4, !dbg !43
}

declare void @llvm.lifetime.start.p0(ptr captures(none))
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg)
declare void @llvm.lifetime.end.p0(ptr captures(none))

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !7, !8, !9, !10}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "a.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"target-abi", !"lp64d"}
!5 = !{i32 6, !"riscv-isa", !6}
!6 = !{!"rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0_zcd1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"}
!7 = !{i32 8, !"PIC Level", i32 2}
!8 = !{i32 7, !"PIE Level", i32 2}
!9 = !{i32 7, !"uwtable", i32 2}
!10 = !{i32 8, !"SmallDataLimit", i32 0}
!14 = !{!"omnipotent char", !15, i64 0}
!15 = !{!"Simple C/C++ TBAA"}
!16 = distinct !DISubprogram(name: "get_i32x4", scope: !17, file: !17, line: 4, type: !18, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !28, keyInstructions: true)
!17 = !DIFile(filename: "a.c", directory: "/")
!18 = !DISubroutineType(types: !19)
!19 = !{!20, !27}
!20 = !DIDerivedType(tag: DW_TAG_typedef, name: "vint32m1_t", file: !17, baseType: !22)
!22 = !DIDerivedType(tag: DW_TAG_typedef, name: "__rvv_int32m1_t", file: !17, baseType: !23)
!23 = !DICompositeType(tag: DW_TAG_array_type, baseType: !24, flags: DIFlagVector, elements: !25)
!24 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!25 = !{!26}
!26 = !DISubrange(lowerBound: 0, upperBound: !DIExpression(DW_OP_bregx, 7202, 0, DW_OP_constu, 4, DW_OP_div, DW_OP_constu, 1, DW_OP_mul, DW_OP_constu, 1, DW_OP_minus))
!27 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !24, size: 64)
!28 = !{!29, !30}
!29 = !DILocalVariable(name: "v", arg: 1, scope: !16, file: !17, line: 4, type: !27)
!30 = !DILocalVariable(name: "r", scope: !16, file: !17, line: 5, type: !20)
!31 = !{!32, !32, i64 0}
!32 = !{!"p1 int", !33, i64 0}
!33 = !{!"any pointer", !14, i64 0}
!34 = !DILocation(line: 4, column: 27, scope: !16)
!35 = !DILocation(line: 5, column: 3, scope: !16)
!36 = !DILocation(line: 5, column: 14, scope: !16)
!37 = !DILocation(line: 6, column: 14, scope: !16)
!38 = !DILocation(line: 6, column: 3, scope: !16, atomGroup: 1, atomRank: 1)
!39 = !DILocation(line: 7, column: 10, scope: !16, atomGroup: 3, atomRank: 2)
!40 = !{!41, !41, i64 0}
!41 = !{!"__rvv_int32m1_t", !14, i64 0}
!42 = !DILocation(line: 8, column: 1, scope: !16)
!43 = !DILocation(line: 7, column: 3, scope: !16, atomGroup: 3, atomRank: 1)
