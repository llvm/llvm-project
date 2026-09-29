; RUN: llc %s --filetype=obj -o - -dwarf-linkage-names=Abstract -add-linkage-names-to-declaration-call-origins=false \
; RUN: | llvm-dwarfdump - | FileCheck %s --check-prefixes=COMMON,DISABLE --implicit-check-not=DW_AT_linkage_name
; RUN: llc %s --filetype=obj -o - -dwarf-linkage-names=Abstract -add-linkage-names-to-declaration-call-origins=true \
; RUN: | llvm-dwarfdump - | FileCheck %s --check-prefixes=COMMON,ENABLE --implicit-check-not=DW_AT_linkage_name

;; Check that -add-linkage-names-to-declaration-call-origins controls whether
;; linkage names are added to declarations referenced by DW_AT_call_origin
;; attributes.
;;
;; $ cat test.cpp
;; void a();
;; __attribute__((optnone))
;; void b() {}
;; void c();
;; extern "C" {
;;   void d();
;; }
;;
;; void e() {
;;   a(); //< Reference declaration DIE (add linkage name).
;;   b(); //< Reference definition DIE (don't add linkage name).
;;   c(); //< Reference definition DIE (don't add linkage name).
;;   d(); //< Reference declaration DIE, but there's no linkage name.
;; }
;;
;; __attribute__((optnone))
;; void c() {}
;; $ clang++ -emit-llvm -S -O1 -g

; COMMON:       DW_TAG_call_site
; ENABLE-NEXT:    DW_AT_call_origin    (0x[[a:[a-z0-9]+]] "_Z1av")
; DISABLE-NEXT:   DW_AT_call_origin    (0x[[a:[a-z0-9]+]] "a")
; COMMON:       DW_TAG_call_site
; COMMON-NEXT:    DW_AT_call_origin     (0x[[b:[a-z0-9]+]] "b")
; COMMON:       DW_TAG_call_site
; COMMON-NEXT:    DW_AT_call_origin     (0x[[c:[a-z0-9]+]] "c")
; COMMON:       DW_TAG_call_site
; COMMON-NEXT:    DW_AT_call_origin     (0x[[d:[a-z0-9]+]] "d")

; COMMON: 0x[[a]]: DW_TAG_subprogram
; COMMON:   DW_AT_name  ("a")
; ENABLE:   DW_AT_linkage_name  ("_Z1av")
; COMMON: 0x[[b]]: DW_TAG_subprogram
; COMMON:   DW_AT_name  ("b")
; COMMON: 0x[[c]]: DW_TAG_subprogram
; COMMON:   DW_AT_name  ("c")
; COMMON: 0x[[d]]: DW_TAG_subprogram
; COMMON:   DW_AT_name  ("d")

target triple = "x86_64-unknown-linux-gnu"

define dso_local void @_Z1ev() local_unnamed_addr !dbg !13 {
entry:
  tail call void @_Z1av(), !dbg !14
  tail call void @_Z1bv(), !dbg !15
  tail call void @_Z1cv(), !dbg !16
  tail call void @d(), !dbg !17
  ret void, !dbg !18
}

define dso_local void @_Z1bv() local_unnamed_addr !dbg !9 {
entry:
  ret void, !dbg !12
}

declare !dbg !19 void @_Z1av() local_unnamed_addr

define dso_local void @_Z1cv() local_unnamed_addr !dbg !20 {
entry:
  ret void, !dbg !21
}
declare !dbg !22 void @d() local_unnamed_addr

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 19.0.0git", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{!"clang version 19.0.0"}
!9 = distinct !DISubprogram(name: "b", linkageName: "_Z1bv", scope: !1, file: !1, line: 3, type: !10, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = !DILocation(line: 3, column: 11, scope: !9)
!13 = distinct !DISubprogram(name: "e", linkageName: "_Z1ev", scope: !1, file: !1, line: 9, type: !10, scopeLine: 9, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!14 = !DILocation(line: 10, column: 3, scope: !13)
!15 = !DILocation(line: 11, column: 3, scope: !13)
!16 = !DILocation(line: 12, column: 3, scope: !13)
!17 = !DILocation(line: 13, column: 3, scope: !13)
!18 = !DILocation(line: 14, column: 1, scope: !13)
!19 = !DISubprogram(name: "a", linkageName: "_Z1av", scope: !1, file: !1, line: 1, type: !10, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!20 = distinct !DISubprogram(name: "c", linkageName: "_Z1cv", scope: !1, file: !1, line: 17, type: !10, scopeLine: 17, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!21 = !DILocation(line: 17, column: 11, scope: !20)
!22 = !DISubprogram(name: "d", scope: !1, file: !1, line: 6, type: !10, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
