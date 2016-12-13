; RUN: not llc -O0 -global-isel -verify-machineinstrs %s -o - 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: llc -O0 -global-isel -global-isel-abort=0 -verify-machineinstrs %s -o - 2>&1 | FileCheck %s --check-prefix=FALLBACK
; RUN: llc -O0 -global-isel -global-isel-abort=2 -verify-machineinstrs %s -o %t.out 2> %t.err
; RUN: FileCheck %s --check-prefix=FALLBACK-WITH-REPORT-OUT < %t.out
; RUN: FileCheck %s --check-prefix=FALLBACK-WITH-REPORT-ERR < %t.err
; This file checks that the fallback path to selection dag works.
; The test is fragile in the sense that it must be updated to expose
; something that fails with global-isel.
; When we cannot produce a test case anymore, that means we can remove
; the fallback path.

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--"

; We use __fixunstfti as the common denominator for __fixunstfti on Linux and
; ___fixunstfti on iOS
; ERROR: Unable to lower arguments
; FALLBACK: ldr q0,
; FALLBACK-NEXT: bl __fixunstfti
;
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for ABIi128
; FALLBACK-WITH-REPORT-OUT-LABEL: ABIi128:
; FALLBACK-WITH-REPORT-OUT: ldr q0,
; FALLBACK-WITH-REPORT-OUT-NEXT: bl __fixunstfti
define i128 @ABIi128(i128 %arg1) {
  %farg1 =       bitcast i128 %arg1 to fp128
  %res = fptoui fp128 %farg1 to i128
  ret i128 %res
}

; It happens that we don't handle ConstantArray instances yet during
; translation. Any other constant would be fine too.

; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for constant
; FALLBACK-WITH-REPORT-OUT-LABEL: constant:
; FALLBACK-WITH-REPORT-OUT: fmov d0, #1.0
define [1 x double] @constant() {
  ret [1 x double] [double 1.0]
}

  ; The key problem here is that we may fail to create an MBB referenced by a
  ; PHI. If so, we cannot complete the G_PHI and mustn't try or bad things
  ; happen.
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for pending_phis
; FALLBACK-WITH-REPORT-OUT-LABEL: pending_phis:
define i32 @pending_phis(i1 %tst, i32 %val, i32* %addr) {
  br i1 %tst, label %true, label %false

end:
  %res = phi i32 [%val, %true], [42, %false]
  ret i32 %res

true:
  store atomic i32 42, i32* %addr seq_cst, align 4
  br label %end

false:
  br label %end

}

  ; General legalizer inability to handle types whose size wasn't a power of 2.
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for odd_type
; FALLBACK-WITH-REPORT-OUT-LABEL: odd_type:
define void @odd_type(i42* %addr) {
  %val42 = load i42, i42* %addr
  ret void
}

  ; RegBankSelect crashed when given invalid mappings, and AArch64's
  ; implementation produce valid-but-nonsense mappings for G_SEQUENCE.
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for sequence_mapping
; FALLBACK-WITH-REPORT-OUT-LABEL: sequence_mapping:
define void @sequence_mapping([2 x i64] %in) {
  ret void
}

  ; Legalizer was asserting when it enountered an unexpected default action.
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for legal_default
; FALLBACK-WITH-REPORT-LABEL: legal_default:
define void @legal_default(i64 %in) {
  insertvalue [2 x i64] undef, i64 %in, 0
  ret void
}

; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for debug_insts
; FALLBACK-WITH-REPORT-LABEL: debug_insts:
define void @debug_insts(i32 %in) #0 !dbg !7 {
entry:
  %in.addr = alloca i32, align 4
  store i32 %in, i32* %in.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %in.addr, metadata !11, metadata !12), !dbg !13
  ret void, !dbg !14
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 4.0.0 (trunk 289075) (llvm/trunk 289080)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "tmp.c", directory: "/Users/tim/llvm/build")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"clang version 4.0.0 (trunk 289075) (llvm/trunk 289080)"}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocalVariable(name: "in", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!12 = !DIExpression()
!13 = !DILocation(line: 1, column: 14, scope: !7)
!14 = !DILocation(line: 2, column: 1, scope: !7)
