; RUN: opt %s -passes='always-inline' -S | FileCheck %s

;; This is the shape the tile compiler emits: the front end inlined `helper`
;; into `kernel`, THEN captured the tile-IR snapshot, so the snapshot (irlayers)
;; sits on the OUTER `kernel` frame -- the instruction head (`helper`) carries
;; no layer of its own. The layer lives on whichever DILocation
;; frame was outermost when the snapshot was taken (here, `kernel`).
;;
;; `kernel` is then inlined into `caller` by the LLVM inliner -- an inline AFTER
;; the snapshot. The `kernel` frame becomes an INNER frame of the deeper chain,
;; and its irlayers must be PRESERVED, not stripped (appendInlinedAt /
;; InlinedAtNode). The NVPTX backend later finds this layer by
;; walking head -> outward to the first layer-bearing frame.

define void @kernel(ptr %p) alwaysinline !dbg !6 {
  store ptr null, ptr %p, align 8, !dbg !9
  ret void, !dbg !8
}

define void @caller(ptr %p) !dbg !12 {
  call void @kernel(ptr %p), !dbg !15
  ret void, !dbg !16
}

; CHECK-LABEL: define void @caller
; CHECK: store ptr null, ptr %p,{{.*}} !dbg ![[INST:[0-9]+]]

;; The inlined store keeps the helper (head) source loc; its inlinedAt now points
;; at the kernel frame.
; CHECK-DAG: ![[INST]] = !DILocation(line: 6, column: 1, scope: ![[HELPER:[0-9]+]], inlinedAt: ![[KFRAME:[0-9]+]])
;; The kernel frame is an INNER frame now (inlinedAt the caller call site) and
;; RETAINS its snapshot layer -- the whole point of the outermost-frame model.
; CHECK-DAG: ![[KFRAME]] = distinct !DILocation(line: 11, column: 1, scope: ![[KERNEL:[0-9]+]], inlinedAt: ![[CS:[0-9]+]], irlayers: ![[LIST:[0-9]+]])
;; The appended outermost frame is the caller's call site, with no layer of its own.
; CHECK-DAG: ![[CS]] = distinct !DILocation(line: 15, column: 1, scope: ![[CALLER:[0-9]+]])
; CHECK-DAG: ![[LIST]] = !DILayerLocList(![[LAYER:[0-9]+]])
; CHECK-DAG: ![[LAYER]] = !DILayerLoc(line: 100, column: 1, file: {{![0-9]+}}, kind: "tile ir")

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "tile", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "kernel.py", directory: "/k")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !5)
!5 = !{null}

;; kernel (outer-frame scope) and the front-end-inlined helper (head scope).
!6 = distinct !DISubprogram(name: "kernel", scope: !1, file: !1, line: 10, type: !4, spFlags: DISPFlagDefinition, unit: !0)
!7 = distinct !DISubprogram(name: "helper", scope: !1, file: !1, line: 5, type: !4, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DILocation(line: 10, column: 1, scope: !6)

;; tile-IR snapshot file + the snapshot layer.
!18 = !DIFile(filename: "kernel.tileir", directory: ".", checksumkind: CSK_MD5, checksum: "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
!19 = !DILayerLoc(line: 100, column: 1, file: !18, kind: "tile ir")
!17 = !DILayerLocList(!19)

;; The kernel (outer) frame carries the snapshot; the helper (head) does not.
!10 = !DILocation(line: 11, column: 1, scope: !6, irlayers: !17)
!9 = !DILocation(line: 6, column: 1, scope: !7, inlinedAt: !10)

;; caller.
!12 = distinct !DISubprogram(name: "caller", scope: !1, file: !1, line: 20, type: !4, spFlags: DISPFlagDefinition, unit: !0)
!15 = !DILocation(line: 15, column: 1, scope: !12)
!16 = !DILocation(line: 16, column: 1, scope: !12)
