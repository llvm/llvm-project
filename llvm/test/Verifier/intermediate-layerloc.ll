; RUN: llvm-as -disable-output %s -o - 2>&1 | FileCheck %s

define void @f(ptr %p) !dbg !5 {
; CHECK: irlayers must be a DILayerLocList
  store i32 0, ptr %p, !dbg !20
; CHECK: layer kind must be a non-null MDString
  store i32 1, ptr %p, !dbg !21
; CHECK: layer file must be a non-null DIFile
  store i32 2, ptr %p, !dbg !22
; CHECK: DILayerLocList must be non-empty
  store i32 3, ptr %p, !dbg !23
; CHECK: DILayerLocList entry must be a DILayerLoc
  store i32 4, ptr %p, !dbg !24
  ret void
}

; CHECK: warning: ignoring invalid debug info

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !{null})
!5 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)

!10 = !DIFile(filename: "t.tileir", directory: "/")
!11 = !DILayerLoc(line: 1, file: !10, kind: "tile ir")

;; irlayers points at a layer rather than a list.
!20 = !DILocation(line: 2, scope: !5, irlayers: !11)

;; An empty kind string parses to a null kind.
!21 = !DILocation(line: 3, scope: !5, irlayers: !30)
!30 = !DILayerLocList(!31)
!31 = !DILayerLoc(line: 1, file: !10, kind: "")

!22 = !DILocation(line: 4, scope: !5, irlayers: !32)
!32 = !DILayerLocList(!33)
!33 = !DILayerLoc(line: 1, file: !4, kind: "tile ir")

!23 = !DILocation(line: 5, scope: !5, irlayers: !34)
!34 = !DILayerLocList()

!24 = !DILocation(line: 6, scope: !5, irlayers: !35)
!35 = !DILayerLocList(!10)
