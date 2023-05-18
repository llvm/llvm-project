; XFAIL: target={{.*}}-aix{{.*}}
; RUN: %llc_dwarf -debugger-tune=lldb -accel-tables=Dwarf -filetype=obj -o %t < %s
; RUN: llvm-dwarfdump -find=_GLOBAL__sub_I__ %t | FileCheck --check-prefix=INFO %s
; RUN: llvm-dwarfdump -debug-names %t | FileCheck --check-prefix=NAMES %s
; RUN: llvm-dwarfdump -debug-names -verify %t | FileCheck --check-prefix=VERIFY %s

; The debug info entry should not have a DW_AT_name, only a DW_AT_linkage_name.
; INFO: DW_TAG_subprogram
; INFO-NOT: DW_AT_name
; INFO:     DW_AT_linkage_name	("_GLOBAL__sub_I__")
; INFO-NOT: DW_AT_name

; The accelerator table should contain only one entry.
; NAMES: Name count: 1
; And it should be the linkage name.
; NAMES: String: 0x{{[0-9a-f]*}} "_GLOBAL__sub_I__"

; Verification should succeed.
; VERIFY: No errors.

define internal void @_GLOBAL__sub_I__() !dbg !7 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 7.0.0 (trunk 329378) (llvm/trunk 329379)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !2)
!1 = !DIFile(filename: "-", directory: "/usr/local/google/home/labath/ll/build/opt")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 7.0.0 (trunk 329378) (llvm/trunk 329379)"}
!7 = distinct !DISubprogram(linkageName: "_GLOBAL__sub_I__", scope: !1, file: !1, type: !8, isLocal: true, isDefinition: true, flags: DIFlagArtificial, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !2)
