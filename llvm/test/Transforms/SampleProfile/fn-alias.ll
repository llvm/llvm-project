; RUN: opt -S %s -passes=sample-profile -sample-profile-file=%S/Inputs/fn-alias.prof | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_ZN1AC1Ei = alias void (ptr, i32), ptr @_ZN1AC2Ei

define void @_ZN1AC2Ei() {
  ret void
}

; CHECK-LABEL: define i32 @main
define i32 @main() #0 !dbg !4 {
; CHECK: call void @_ZN1AC1Ei
; CHECK-SAME: !prof ![[PROF:[0-9]+]]
  call void @_ZN1AC1Ei(ptr null, i32 0), !dbg !7
  ret i32 0
}

attributes #0 = { "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 22.0.0git (https://github.com/llvm/llvm-project 2a02d57efb22956665d91b68de6b3f8d58db9cda)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2, globals: !2, imports: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "main.cpp", directory: "/", checksumkind: CSK_MD5, checksum: "0f77ac0d94ea63dd55c86ed20b80e841")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "main", scope: !5, file: !5, line: 20, type: !6, scopeLine: 20, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!5 = !DIFile(filename: "./main.cpp", directory: "/", checksumkind: CSK_MD5, checksum: "0f77ac0d94ea63dd55c86ed20b80e841")
!6 = distinct !DISubroutineType(types: !2)
!7 = !DILocation(line: 22, column: 7, scope: !8)
!8 = distinct !DILexicalBlock(scope: !9, file: !5, line: 21, column: 47)
!9 = distinct !DILexicalBlock(scope: !10, file: !5, line: 21, column: 3)
!10 = distinct !DILexicalBlock(scope: !4, file: !5, line: 21, column: 3)

; CHECK: ![[PROF]] = !{!"VP", i32 0, i64 212088, i64 6296505300821684249, i64 212088}
