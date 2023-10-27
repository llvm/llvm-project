; REQUIRES: x86-registered-target
; RUN: llc < %s | FileCheck %s

; Repro for issue https://reviews.llvm.org/D149367#4619121
; Validates that `indirect ptr null` and a jump table can be used in the same function.

; Verify branch labels match what's in the CodeView
; CHECK:            .Ltmp2:
; CHECK-NEXT:       jmpq    *%{{.*}}

; Verify jump table have the same entry size, base offset and shift as what's in the CodeView
; CHECK:          {{\.?}}LJTI0_0:
; CHECK-NEXT:     .long   .LBB0_[[#]]-.LJTI0_0

; Verify CodeView
; CHECK:          .short	4441          # Record kind: S_ARMSWITCHTABLE
; CHECK-NEXT:     .secrel32	.LJTI0_0    # Base offset
; CHECK-NEXT:     .secidx	.LJTI0_0      # Base section index
; CHECK-NEXT:     .short	4             # Switch type
; CHECK-NEXT:     .secrel32	.Ltmp2      # Branch offset
; CHECK-NEXT:     .secrel32	.LJTI0_0    # Table offset
; CHECK-NEXT:     .secidx	.Ltmp2        # Branch section index
; CHECK-NEXT:     .secidx	.LJTI0_0      # Table section index
; CHECK-NEXT:     .long	4               # Entries count
; CHECK-NOT:      .short	4441          # Record kind: S_ARMSWITCHTABLE

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.34.0"

define i32 @f() !dbg !5 {
entry:
  indirectbr ptr null, [label %BC_SUCCEED], !dbg !11

BC_SUCCEED:                                       ; preds = %entry
  %0 = lshr i64 0, 0
  switch i64 %0, label %sw.default.i.i2445 [
    i64 3, label %sw.bb15.i.i
    i64 1, label %sw.bb7.i.i
    i64 2, label %sw.bb11.i.i2444
    i64 0, label %sw.bb3.i.i
  ]

sw.bb3.i.i:                                       ; preds = %BC_SUCCEED
  ret i32 0

sw.bb7.i.i:                                       ; preds = %BC_SUCCEED
  ret i32 0

sw.bb11.i.i2444:                                  ; preds = %BC_SUCCEED
  ret i32 0

sw.bb15.i.i:                                      ; preds = %BC_SUCCEED
  ret i32 0

sw.default.i.i2445:                               ; preds = %BC_SUCCEED
  ret i32 0
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2, imports: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "../../v8/src/regexp\\regexp-interpreter.cc", directory: ".", checksumkind: CSK_MD5, checksum: "ddba353f72137fb1d64b5fc8ee071a9c")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "f", linkageName: "f", scope: !7, file: !6, line: 386, type: !10, scopeLine: 391, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, templateParams: !2, retainedNodes: !2)
!6 = !DIFile(filename: "../../v8/src/regexp/regexp-interpreter.cc", directory: ".", checksumkind: CSK_MD5, checksum: "ddba353f72137fb1d64b5fc8ee071a9c")
!7 = !DINamespace(scope: !8)
!8 = !DINamespace(name: "internal", scope: !9)
!9 = !DINamespace(name: "v8", scope: null)
!10 = distinct !DISubroutineType(types: !2)
!11 = !DILocation(line: 1, scope: !5)