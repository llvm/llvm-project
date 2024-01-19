; RUN: opt -passes=sroa -S -o - %s \
; RUN: | FileCheck %s --implicit-check-not="call void @llvm.dbg"
; RUN: opt --try-experimental-debuginfo-iterators -passes=sroa -S -o - %s \
; RUN: | FileCheck %s --implicit-check-not="call void @llvm.dbg"
;
;; Based on llvm/test/DebugInfo/ARM/sroa-complex.ll
;; generated from:
;; $ cat test.c
;; void f(_Complex double c) { c = 0; }
;; $ clang test.c -g -O2 -c -Xclang -disable-llvm-passes -S \
;;     -emit-llvm -o - --target="thumbv7-apple-unknown"
;;
;; Commented out some parts of the function that are not relevant to the test.
;;
;; Check that a split store gets dbg.assigns fragments. Ensure that only the
;; value-expression gets fragment info; that the address-expression remains
;; untouched.

;; dbg.assigns for the split (then promoted) stores.
; CHECK: %c.coerce.fca.0.extract = extractvalue [2 x i64] %c.coerce, 0
; CHECK: %c.coerce.fca.1.extract = extractvalue [2 x i64] %c.coerce, 1
; CHECK: call void @llvm.dbg.value(metadata i64 %c.coerce.fca.0.extract,{{.+}}, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 64))
; CHECK: call void @llvm.dbg.value(metadata i64 %c.coerce.fca.1.extract,{{.+}}, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 64))

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7-apple-unknown"

define dso_local arm_aapcscc void @f([2 x i64] %c.coerce) #0 !dbg !8 {
entry:
  %c = alloca { double, double }, align 8, !DIAssignID !14
  call void @llvm.dbg.assign(metadata i1 undef, metadata !13, metadata !DIExpression(), metadata !14, metadata ptr %c, metadata !DIExpression()), !dbg !15
  %0 = bitcast ptr %c to [2 x i64]*
  store [2 x i64] %c.coerce, [2 x i64]* %0, align 8, !DIAssignID !16
  call void @llvm.dbg.assign(metadata [2 x i64] %c.coerce, metadata !13, metadata !DIExpression(), metadata !16, metadata [2 x i64]* %0, metadata !DIExpression()), !dbg !15
  ; --- The rest of this function isn't useful for the test ---
  ;%c.realp = getelementptr inbounds { double, double }, ptr %c, i32 0, i32 0, !dbg !17
  ;%c.imagp = getelementptr inbounds { double, double }, ptr %c, i32 0, i32 1, !dbg !17
  ;store double 0.000000e+00, ptr %c.realp, align 8, !dbg !17, !DIAssignID !18
  ;call void @llvm.dbg.assign(metadata double 0.000000e+00, metadata !13, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 64), metadata !18, metadata ptr %c.realp, metadata !DIExpression()), !dbg !15
  ;store double 0.000000e+00, ptr %c.imagp, align 8, !dbg !17, !DIAssignID !19
  ;call void @llvm.dbg.assign(metadata double 0.000000e+00, metadata !13, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 64), metadata !19, metadata ptr %c.imagp, metadata !DIExpression()), !dbg !15
  ret void, !dbg !20
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6, !1000}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 1, !"min_enum_size", i32 4}
!7 = !{!"clang version 12.0.0"}
!8 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 2, type: !9, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11}
!11 = !DIBasicType(name: "complex", size: 128, encoding: DW_ATE_complex_float)
!12 = !{!13}
!13 = !DILocalVariable(name: "c", arg: 1, scope: !8, file: !1, line: 2, type: !11)
!14 = distinct !DIAssignID()
!15 = !DILocation(line: 0, scope: !8)
!16 = distinct !DIAssignID()
!17 = !DILocation(line: 2, column: 31, scope: !8)
!18 = distinct !DIAssignID()
!19 = distinct !DIAssignID()
!20 = !DILocation(line: 2, column: 36, scope: !8)
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
