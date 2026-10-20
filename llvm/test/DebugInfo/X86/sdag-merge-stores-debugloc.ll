; RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs -stop-after=finalize-isel -o - %s | FileCheck %s

;; Check debug locations on merged 64-bit loads and stores.
;; Locations from inl_a and inl_b merge to line 0 in their common caller.
;; Identical load locations are retained. The rotate keeps the location of
;; the original load at %s.

; Capture input locations from the IR printed before the machine functions.
; CHECK-LABEL: define void @store_constants_inlined_siblings(
; CHECK-SAME: !dbg ![[STORE_SCOPE:[0-9]+]]
; CHECK-LABEL: define void @copy_loads_inlined_siblings(
; CHECK-SAME: !dbg ![[COPY_SCOPE:[0-9]+]]
; CHECK: load i32, ptr %s, align 4, !dbg ![[COPY_LOAD_LOC:[0-9]+]]
; CHECK-LABEL: define void @rotate_loads_inlined_siblings(
; CHECK-SAME: !dbg ![[ROTATE_SCOPE:[0-9]+]]
; CHECK: load i32, ptr %s, align 4, !dbg ![[ROTATE_LOAD_LOC:[0-9]+]]

;; The two i32 stores come from inlined inl_a and inl_b.
;; The merged 64-bit store must use line 0 in the outer function,
;; without retaining either inlined function's frame.
; CHECK-LABEL: name: store_constants_inlined_siblings
; CHECK: MOV64mi32 {{.*}}, debug-location !DILocation(line: 0, scope: ![[STORE_SCOPE]]) :: (store (s64)
define void @store_constants_inlined_siblings(ptr %p) !dbg !10 {
  store i32 0, ptr %p, align 4, !dbg !13
  %q = getelementptr i32, ptr %p, i64 1
  store i32 0, ptr %q, align 4, !dbg !14
  ret void
}
!10 = distinct !DISubprogram(name: "store_constants_inlined_siblings", scope: !1, file: !1, line: 1, type: !2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !DILocation(line: 30, scope: !10)
!12 = !DILocation(line: 40, scope: !10)
!13 = !DILocation(line: 10, scope: !5, inlinedAt: !11)
!14 = !DILocation(line: 20, scope: !6, inlinedAt: !12)

;; Both loads have the same location in inlined inl_a; the stores come from
;; inl_a and inl_b. The merged 64-bit load must keep inl_a's location,
;; while the merged 64-bit store must use line 0 in the outer function.
; CHECK-LABEL: name: copy_loads_inlined_siblings
; CHECK: MOV64rm {{.*}}, debug-location ![[COPY_LOAD_LOC]] :: (load (s64)
; CHECK: MOV64mr {{.*}}, debug-location !DILocation(line: 0, scope: ![[COPY_SCOPE]]) :: (store (s64)
define void @copy_loads_inlined_siblings(ptr noalias %d, ptr noalias %s) !dbg !20 {
  %s1 = getelementptr i32, ptr %s, i64 1
  %d1 = getelementptr i32, ptr %d, i64 1
  %a = load i32, ptr %s, align 4, !dbg !23
  %b = load i32, ptr %s1, align 4, !dbg !23
  store i32 %a, ptr %d, align 4, !dbg !23
  store i32 %b, ptr %d1, align 4, !dbg !24
  ret void
}
!20 = distinct !DISubprogram(name: "copy_loads_inlined_siblings", scope: !1, file: !1, line: 1, type: !2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!21 = !DILocation(line: 30, scope: !20)
!22 = !DILocation(line: 40, scope: !20)
!23 = !DILocation(line: 10, scope: !5, inlinedAt: !21)
!24 = !DILocation(line: 20, scope: !6, inlinedAt: !22)

;; The loads from inlined inl_a and inl_b are stored in reverse order.
;; Expect a 64-bit load, rotate by 32, and 64-bit store. The load/store
;; use outer line 0; the rotate keeps the location of the original load at %s.
; CHECK-LABEL: name: rotate_loads_inlined_siblings
; CHECK: MOV64rm {{.*}}, debug-location !DILocation(line: 0, scope: ![[ROTATE_SCOPE]]) :: (load (s64)
; CHECK: {{(ROL|ROR)64ri}} {{.*}}, 32, {{.*}}debug-location ![[ROTATE_LOAD_LOC]]{{$}}
; CHECK: MOV64mr {{.*}}, debug-location !DILocation(line: 0, scope: ![[ROTATE_SCOPE]]) :: (store (s64)
define void @rotate_loads_inlined_siblings(ptr noalias %d, ptr noalias %s) !dbg !30 {
  %s1 = getelementptr i32, ptr %s, i64 1
  %d1 = getelementptr i32, ptr %d, i64 1
  %a = load i32, ptr %s, align 4, !dbg !33
  %b = load i32, ptr %s1, align 4, !dbg !34
  store i32 %b, ptr %d, align 4, !dbg !34
  store i32 %a, ptr %d1, align 4, !dbg !33
  ret void
}
!30 = distinct !DISubprogram(name: "rotate_loads_inlined_siblings", scope: !1, file: !1, line: 1, type: !2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!31 = !DILocation(line: 30, scope: !30)
!32 = !DILocation(line: 40, scope: !30)
!33 = !DILocation(line: 10, scope: !5, inlinedAt: !31)
!34 = !DILocation(line: 20, scope: !6, inlinedAt: !32)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}
!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "test", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "store-merge.c", directory: "/")
!2 = !DISubroutineType(types: !4)
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{}
!5 = distinct !DISubprogram(name: "inl_a", scope: !1, file: !1, line: 1, type: !2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = distinct !DISubprogram(name: "inl_b", scope: !1, file: !1, line: 2, type: !2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
