; RUN: %llc_dwarf -split-dwarf-file=foo.dwo  %s -filetype=obj -o - | llvm-dwarfdump -debug-info - | FileCheck %s

; DW_AT_LLVM_sysroot goes into the .dwo, not in the skeleton.

; CHECK: DW_TAG_skeleton_unit
; CHECK-NOT: DW_AT_LLVM_sysroot
; CHECK-NOT: DW_AT_LLVM_sdk
; CHECK: DW_TAG_compile_unit
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_LLVM_sysroot ("/opt/clang-root")
; CHECK: DW_AT_APPLE_sdk ("Linux.sdk")

target triple = "x86_64-pc-linux"

declare void @_Z2f1v()

; Function Attrs: noinline norecurse uwtable
define i32 @main() !dbg !9 {
entry:
  ret i32 0, !dbg !18
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6, !7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, emissionKind: FullDebug, enums: !2, sysroot: "/opt/clang-root", sdk: "Linux.sdk")
!1 = !DIFile(filename: "a.c", directory: "/")
!2 = !{}
!6 = !{i32 2, !"Dwarf Version", i32 5}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 2, type: !10, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!18 = !DILocation(line: 4, column: 1, scope: !9)
