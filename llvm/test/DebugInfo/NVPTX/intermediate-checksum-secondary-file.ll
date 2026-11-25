; RUN: llc -O0 < %s -mtriple=nvptx64-nvidia-cuda | FileCheck %s
;;
;; A secondary .file is named by the checksum its DIFile carries, not by a hash
;; of the filename. The checksum below is deliberately not
;; MD5("intermediate.ir"), so hashing the filename would fail the test.

define dso_local void @test_kernel(ptr noundef %v) #0 !dbg !8 {
entry:
  %v.addr = alloca ptr, align 8
  store ptr %v, ptr %v.addr, align 8, !dbg !20
  ret void, !dbg !21
}

attributes #0 = { noinline optnone }

;; The .file that .loc_intermediate refers to is named by the checksum, under
;; the DIFile's directory.
; CHECK: .loc_intermediate [[INTFILE:[0-9]+]] 42 5
; CHECK: .file [[INTFILE]] ".{{/|\\\\}}cafebabecafebabecafebabecafebabe"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cu", directory: "/test")
!2 = !{i32 7, !"Dwarf Version", i32 2}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "test_kernel", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DISubroutineType(types: !10)
!10 = !{null}

;; The checksum is not a hash of the filename.
!14 = !DIFile(filename: "intermediate.ir", directory: ".", checksumkind: CSK_MD5, checksum: "cafebabecafebabecafebabecafebabe", source: "intermediate ir source")
!30 = !DILayerLocList(!31)
!31 = !DILayerLoc(line: 42, column: 5, file: !14, kind: "intermediate ir")

!20 = !DILocation(line: 2, column: 5, scope: !8, irlayers: !30)
!21 = !DILocation(line: 3, column: 1, scope: !8)
