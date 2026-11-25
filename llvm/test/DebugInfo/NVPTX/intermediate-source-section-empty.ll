; RUN: llc -O0 < %s -mtriple=nvptx64-nvidia-cuda | FileCheck %s
;;
;; A layer whose DIFile has no source: text is dropped, because a
;; .loc_intermediate is only meaningful with the source it points into and the
;; consumer cannot resolve a reference without it. With no layer left, no source
;; section is emitted either, not even an empty one.

;; PTX is still produced.
; CHECK: .target sm_{{[0-9]+}}
;; The layer leaves no trace: no .loc_intermediate, no secondary .file and no
;; source section.
; CHECK-NOT: .loc_intermediate
; CHECK-NOT: cccccccccccccccccccccccccccccccc
; CHECK-NOT: .nv_intermediate_source_section

define dso_local void @no_intermediate_source(ptr noundef %v) #0 !dbg !8 {
entry:
  %v.addr = alloca ptr, align 8
  store ptr %v, ptr %v.addr, align 8, !dbg !11
  ret void, !dbg !12
}

attributes #0 = { noinline optnone "target-cpu"="sm_75" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cu", directory: "/test")
!2 = !{i32 7, !"Dwarf Version", i32 2}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "no_intermediate_source", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DISubroutineType(types: !10)
!10 = !{null}

;; Intermediate file with a checksum but no source: text.
!4 = !DIFile(filename: "unused.ir", directory: ".", checksumkind: CSK_MD5, checksum: "cccccccccccccccccccccccccccccccc")
!5 = !DILayerLoc(line: 100, column: 10, file: !4, kind: "intermediate ir")
!6 = !DILayerLocList(!5)

;; The store carries the layer; the ret has none.
!11 = !DILocation(line: 2, column: 5, scope: !8, irlayers: !6)
!12 = !DILocation(line: 4, column: 1, scope: !8)
