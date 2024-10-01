; RUN: llc -filetype=obj -mtriple i686-pc-windows-msvc %s -o %t.o
; RUN: llvm-pdbutil dump %t.o -symbols -types | FileCheck %s

; CHECK: [[INFO_IDX:0x[^ ]*]] | LF_BUILDINFO
; CHECK-NEXT:          0x{{.*}}: `D:\src\scopes\clang`
; CHECK-NEXT:          0x{{.*}}: ``
; CHECK-NEXT:          0x{{.*}}: `D:\src\scopes\foo.cpp`
; CHECK-NEXT:          0x{{.*}}: ``
; CHECK-NEXT:          0x{{.*}}: ``

; CHECK: {{.*}} | S_BUILDINFO [size = 8] BuildId = `[[INFO_IDX]]`

; ModuleID = 'D:\src\scopes\foo.cpp'
source_filename = "D:\5Csrc\5Cscopes\5Cfoo.cpp"
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc19.0.23918"

define i32 @"?foo@@YAHXZ"() !dbg !10 {
entry:
  ret i32 42, !dbg !14
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 4.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
; One .debug$S section should contain an S_COMPILE3 record that identifies the
; source language and the version of the compiler based on the DICompileUnit.
!1 = !DIFile(filename: "D:\5Csrc\5Cscopes\5Cfoo.cpp", directory: "D:\5Csrc\5Cscopes\5Cclang")
!2 = !{}
!7 = !{i32 2, !"CodeView", i32 1}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{!"clang version 4.0.0 "}
!10 = distinct !DISubprogram(name: "foo", linkageName: "\01?foo@@YAHXZ", scope: !1, file: !1, line: 1, type: !11, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!11 = !DISubroutineType(types: !12)
!12 = !{!13}
!13 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!14 = !DILocation(line: 2, scope: !10)
