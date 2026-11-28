; RUN: llc -O0 < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_70 -mattr=+ptx72 \
; RUN:   | FileCheck %s

;; Consecutive instructions that share a layer list get one .loc_intermediate:
;; the intermediate position is already current, so repeating it would be
;; redundant. This holds whether they share a DILocation (add, mul) or use
;; different ones with the same source position and list (ret).

target triple = "nvptx64-nvidia-cuda"

define i32 @same_list(i32 %a, i32 %b) !dbg !5 {
  %1 = add i32 %a, %b, !dbg !20
  %2 = mul i32 %1, 3,  !dbg !20
  ret i32 %2,          !dbg !21
}

; CHECK-LABEL: same_list
; CHECK:      .loc 1 10 5
; CHECK-NEXT: .loc_intermediate {{[0-9]+}} 100 1
; CHECK-NOT:  .loc
; CHECK:      ret;

!llvm.dbg.cu                    = !{!2}
!llvm.module.flags              = !{!0, !1}

!0 = !{i32 2, !"Dwarf Version", i32 2}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3,
                              emissionKind: DebugDirectivesOnly)
!3 = !DIFile(filename: "demo.c", directory: "/tmp")
!4 = !DISubroutineType(types: !{})
!5 = distinct !DISubprogram(name: "same_list", scope: !3, file: !3,
                            line: 1, type: !4, scopeLine: 1,
                            spFlags: DISPFlagDefinition, unit: !2)

!10 = !DIFile(filename: "demo.ir", directory: "/tmp", checksumkind: CSK_MD5, checksum: "dddddddddddddddddddddddddddddddd", source: "demo ir text")

!110 = !DILayerLoc(line: 100, column: 1, file: !10, kind: "intermediate ir")
!120 = !DILayerLocList(!110)

;; Two DILocations with the same source position and the same layer list.
!20 = !DILocation(line: 10, column: 5, scope: !5, irlayers: !120)
!21 = !DILocation(line: 10, column: 5, scope: !5, irlayers: !120, isImplicitCode: true)
