; REQUIRES: aarch64-registered-target

; This test needs to be target specific due to the cost estimate in the output.

; RUN: opt -passes=lower-matrix-intrinsics -pass-remarks=lower-matrix-intrinsics -mtriple=arm64-apple-iphoneos -S < %s 2>&1 | FileCheck  %s

; Test the propagation of matrix expressions along to inlined-at chain. The IR
; in the test roughly corresponds to the C++ code below, with the IR containing
; references to a few more functions.

; matrix.h
; template <typename Ty, unsigned R, unsigned C>
; struct Matrix {
;   using matrix_t = Ty __attribute__((matrix_type(R, C)));
;
;   matrix_t value;
; };
;
; ; add.h
; template <typename Ty, unsigned R, unsigned C>
; Matrix<Ty, R, C> add(Matrix<Ty, R, C> M1, Matrix<Ty, R, C> M2) {
;   Matrix<Ty, R, C> Result;
;   Result.value = __builtin_matrix_add(M1.value, M2.value);
;   return Result;
; }
;
; load.h:
; template <typename Ty, unsigned R, unsigned C>
; Matrix<Ty, R, C> load(Ty *Ptr) {
;   Matrix<Ty, R, C> Result;
;   Result.value = *reinterpret_cast <typename Matrix<Ty, R, C>::matrix_t *>(Ptr);
;   return Result;
; }
;
; store.h:
; template <typename Ty, unsigned R, unsigned C>
; void store(Matrix<Ty, R, C> M1, Ty *Ptr) {
;   *reinterpret_cast<typename decltype(M1)::matrix_t *>(Ptr) = M1.value;
; }
;
; toplevel.cpp
; void test(double *A, double *B, double *C) {
;   store(add(load<double, 3, 5>(A), load<double, 3, 5>(B)), C);
; }
;

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "aarch64-apple-ios"

; CHECK-LABEL: remark: load.h:41:43: Lowered with 0 stores, 10 loads, 0 compute ops, 0 exposed transposes
; CHECK-NEXT:  load(addr %A)

; CHECK-LABEL: remark: load.h:41:43: Lowered with 0 stores, 10 loads, 0 compute ops, 0 exposed transposes
; CHECK-NEXT:  column.major.load.3x5.double(addr %B, 5)

; CHECK-LABEL: remark: load.h:41:11: Lowered with 0 stores, 1 loads, 0 compute ops, 0 exposed transposes
; CHECK-NEXT: load(addr %D)

; CHECK-LABEL: remark: assign.h:32:43: Lowered with 0 stores, 10 loads, 0 compute ops, 0 exposed transposes
; CHECK-NEXT:  load(addr %A)

; CHECK-LABEL: remark: assign.h:32:43: Lowered with 0 stores, 10 loads, 0 compute ops, 0 exposed transposes
; CHECK-NEXT:  column.major.load.3x5.double(addr %B, 5)

; CHECK-LABEL: remark: toplevel.c:410:0: Lowered with 10 stores, 20 loads, 10 compute ops, 0 exposed transposes
; CHECK-NEXT:  store(
; CHECK-NEXT:   fadd(
; CHECK-NEXT:    load(addr %A),
; CHECK-NEXT:    column.major.load.3x5.double(addr %B, 5)),
; CHECK-NEXT:   addr %C)

; CHECK-LABEL: remark: toplevel.c:510:0: Lowered with 2 stores, 1 loads, 4 compute ops, 1 exposed transposes
; CHECK-NEXT:  store(
; CHECK-NEXT:   transpose.2x1.float(load(addr %D)),
; CHECK-NEXT:   addr %D)

; CHECK-LABEL: remark: add.h:66:11: Lowered with 0 stores, 0 loads, 10 compute ops, 0 exposed transposes
; CHECK-NEXT:  fadd(
; CHECK-NEXT:   addr %A,
; CHECK-NEXT:   scalar)

; CHECK-LABEL: remark: store.h:10:11: Lowered with 10 stores, 0 loads, 0 compute ops, 0 exposed transposes
; CHECK-NEXT:  store(
; CHECK-NEXT:   scalar,
; CHECK-NEXT:   addr %C)

; CHECK-LABEL: remark: store.h:66:11: Lowered with 2 stores, 0 loads, 0 compute ops, 0 exposed transposes
; CHECK-NEXT:  store(
; CHECK-NEXT:  scalar,
; CHECK-NEXT:  addr %D)

; CHECK-LABEL: remark: transpose.h:13:11: Lowered with 0 stores, 0 loads, 4 compute ops, 1 exposed transposes
; CHECK-NEXT:  transpose.2x1.float(addr %D)

define void @toplevel(ptr %A, ptr %B, ptr %C, ptr %D) !dbg !16 {
entry:
  %a = load <15 x double>, ptr %A, align 16, !dbg !3791
  %b = call <15 x double> @llvm.matrix.column.major.load(ptr %B, i64 5, i1 false, i32 3, i32 5), !dbg !3793
  %c  = fadd <15 x double> %a, %b, !dbg !100
  store <15 x double> %c, ptr %C, align 16, !dbg !102

  %load = load <2 x float>, ptr %D, !dbg !104
  %t1 = call <2 x float> @llvm.matrix.transpose(<2 x float> %load, i32 2, i32 1), !dbg !106
  store <2 x float> %t1, ptr %D, !dbg !108
  ret void
}

declare <15 x double> @llvm.matrix.column.major.load(ptr, i64, i1, i32, i32)
declare <2 x float> @llvm.matrix.transpose(<2 x float>, i32, i32)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "load.h", directory: "/test")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "load_fn", scope: !1, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)
!17 = !DIFile(filename: "toplevel.c", directory: "/test")
!16 = distinct !DISubprogram(name: "toplevel", scope: !1, file: !17, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)
!18 = !DIFile(filename: "assign.h", directory: "/test")
!19 = distinct !DISubprogram(name: "assign", scope: !1, file: !18, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)

!20 = !DIFile(filename: "add.h", directory: "/test")
!21 = distinct !DISubprogram(name: "add_fn", scope: !1, file: !20, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)

!22 = !DIFile(filename: "store.h", directory: "/test")
!23 = distinct !DISubprogram(name: "store_fn", scope: !1, file: !22, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)

!24 = !DIFile(filename: "transpose.h", directory: "/test")
!25 = distinct !DISubprogram(name: "transpose", scope: !1, file: !24, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)


!6 = !DISubroutineType(types: !7)
!7 = !{null, !8, !8, !11}
!8 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !9)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 32, align: 32)
!10 = !DIBasicType(name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!11 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!12 = !{!13}
!13 = !DILocalVariable(name: "a", arg: 1, scope: !5, file: !1, line: 1, type: !8)
!14 = !DILocation(line: 1, column: 27, scope: !5)

!3791 = !DILocation(line: 41, column: 43, scope: !5, inlinedAt: !3795)
!3792 = !DILocation(line: 405, column: 3, scope: !16)
!3793 = !DILocation(line: 41, column: 43, scope: !5, inlinedAt: !3796)
!3794 = !DILocation(line: 406, column: 11, scope: !16)
!3795 = !DILocation(line: 32, column: 43, scope: !19, inlinedAt: !3792)
!3796 = !DILocation(line: 32, column: 43, scope: !19, inlinedAt: !3794)

!100 = !DILocation(line: 66, column: 11, scope: !21, inlinedAt: !101)
!101 = !DILocation(line: 410, column: 11, scope: !16)

!102 = !DILocation(line: 10, column: 11, scope: !23, inlinedAt: !103)
!103 = !DILocation(line: 410, column: 0, scope: !16)

!104 = !DILocation(line: 41, column: 11, scope: !5, inlinedAt: !101)
!105 = !DILocation(line: 500, column: 11, scope: !16)

!106 = !DILocation(line: 13, column: 11, scope: !25, inlinedAt: !101)
!107 = !DILocation(line: 510, column: 11, scope: !16)

!108 = !DILocation(line: 66, column: 11, scope: !23, inlinedAt: !109)
!109 = !DILocation(line: 510, column: 0, scope: !16)
