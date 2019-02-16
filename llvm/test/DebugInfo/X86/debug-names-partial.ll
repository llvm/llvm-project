; REQUIRES: object-emission

; Verify that DWARF v5 debug_names omit names from CUs that opt-out.
; RUN: llc -mtriple x86_64-pc-linux -filetype=obj < %s \
; RUN:   | llvm-dwarfdump -debug-info -debug-names - | FileCheck %s


; Check that the one CU that is indexed has a non-zero.
; Avoid checking for a specific offset to make the test more resilient.
; CHECK: [[CU1OFF:0x00000000]]: Compile Unit:
; CHECK: [[CU2OFF:0x[0-9a-f]{8}]]: Compile Unit:
; CHECK: [[CU3OFF:0x[0-9a-f]{8}]]: Compile Unit:

; CHECK: CU count: 2
; CHECK: Compilation Unit offsets [
; CHECK-NEXT: CU[0]: [[CU1OFF]]
; CHECK-NEXT: CU[1]: [[CU3OFF]]
; CHECK-NEXT: ]
; CHECK-NOT: DW_IDX_compile_unit: 0x02
; CHECK: String: {{.*}} "f3"
; CHECK-NOT: DW_IDX_compile_unit
; CHECK: DW_IDX_compile_unit: 0x01

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @_Z2f1v() !dbg !11 {
entry:
  ret void, !dbg !14
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @_Z2f2v() !dbg !15 {
entry:
  ret void, !dbg !16
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @_Z2f3v() !dbg !17 {
entry:
  ret void, !dbg !18
}

!llvm.dbg.cu = !{!0, !3, !5}
!llvm.ident = !{!7, !7, !7}
!llvm.module.flags = !{!8, !9, !10}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 8.0.0 (trunk 340586) (llvm/trunk 340588)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: Default)
!1 = !DIFile(filename: "f1.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch", checksumkind: CSK_MD5, checksum: "5cf4a85ae773dd04a42282b1a708a179")
!2 = !{}
!3 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "clang version 8.0.0 (trunk 340586) (llvm/trunk 340588)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!4 = !DIFile(filename: "f2.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch", checksumkind: CSK_MD5, checksum: "17efa328ddcbb22a3043feeec3190783")
!5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "clang version 8.0.0 (trunk 340586) (llvm/trunk 340588)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: Default)
!6 = !DIFile(filename: "f3.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch", checksumkind: CSK_MD5, checksum: "73ed062a3b287e0193c695c550d2cef2")
!7 = !{!"clang version 8.0.0 (trunk 340586) (llvm/trunk 340588)"}
!8 = !{i32 2, !"Dwarf Version", i32 5}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = distinct !DISubprogram(name: "f1", linkageName: "_Z2f1v", scope: !1, file: !1, line: 1, type: !12, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!12 = !DISubroutineType(types: !13)
!13 = !{null}
!14 = !DILocation(line: 2, column: 1, scope: !11)
!15 = distinct !DISubprogram(name: "f2", linkageName: "_Z2f2v", scope: !4, file: !4, line: 1, type: !12, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !3, retainedNodes: !2)
!16 = !DILocation(line: 2, column: 1, scope: !15)
!17 = distinct !DISubprogram(name: "f3", linkageName: "_Z2f3v", scope: !6, file: !6, line: 1, type: !12, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !5, retainedNodes: !2)
!18 = !DILocation(line: 2, column: 1, scope: !17)
