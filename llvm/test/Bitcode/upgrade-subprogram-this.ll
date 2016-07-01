; RUN: llvm-dis < %s.bc | FileCheck %s
; RUN: verify-uselistorder < %s.bc

; Test upgrading from bitcode without a this adjustment record. It will fill in
; an implicit zero thisAdjustment, so it will not be present in the output.

; CHECK: DISubprogram(name: "f",
; CHECK-NOT: thisAdjustment
; CHECK-SAME: ){{$}}

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.23918"

; Function Attrs: nounwind uwtable
define void @f() !dbg !7 {
entry:
  ret void, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!2 = !{}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!7 = distinct !DISubprogram(name: "f", linkageName: "\01?f@@YAXXZ", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocation(line: 1, column: 11, scope: !7)
