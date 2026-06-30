; REQUIRES: x86_64-linux
; REQUIRES: asserts
; RUN: llvm-profdata merge --sample --extbinary %S/Inputs/stale-profile-lcs-anchor-overwrite.prof -o %t.prof
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%t.prof --salvage-stale-profile --salvage-unused-profile -S --debug-only=sample-profile,sample-profile-matcher,sample-profile-impl 2>&1 | FileCheck %s

; CHECK: Function:_Z3barv matches profile:_Z3bari
; CHECK: The functions _Z3barv(IR) and _Z3barl(Profile) share the same base name: bar.
; CHECK: Function:_Z3barv matches profile:_Z3barl
; CHECK: The functions _Z3barPv(IR) and _Z3barl(Profile) share the same base name: bar.
; CHECK: Function:_Z3barPv matches profile:_Z3barl

; CHECK: Run stale profile matching for _Z3barPv
; CHECK: The functions _Z6calleePv(IR) and _Z6calleei(Profile) share the same base name: callee.
; CHECK: Function:_Z6calleePv matches profile:_Z6calleei
; CHECK: Location is matched from 1 to 1
; CHECK: Callsite with callee:_Z6calleePv is matched from 2 to 3
; CHECK: Run stale profile matching for _Z3barv
; CHECK: Run stale profile matching for _Z6calleePv
; CHECK: Function processing order:
; CHECK: _Z3foov
; CHECK: _Z3barPv
; CHECK: _Z6calleePv
; CHECK: _Z3barv


target triple = "x86_64-linux-gnu"

define dso_local noundef ptr @_Z6calleePv(ptr noundef %ptr) #0 !dbg !15 {
entry:
  %ptr.addr = alloca ptr, align 8
  store ptr %ptr, ptr %ptr.addr, align 8
    #dbg_declare(ptr %ptr.addr, !20, !DIExpression(), !21)
  call void @llvm.pseudoprobe(i64 7108221232740920931, i64 1, i32 0, i64 -1), !dbg !22
  ret ptr null, !dbg !22
}

define dso_local void @_Z3barv() #0 !dbg !23 {
entry:
  call void @llvm.pseudoprobe(i64 -1069303473483922844, i64 1, i32 0, i64 -1), !dbg !24
  ret void, !dbg !24
}

define dso_local noundef ptr @_Z3barPv(ptr noundef %ptr) #0 !dbg !25 {
entry:
  %ptr.addr = alloca ptr, align 8
  store ptr %ptr, ptr %ptr.addr, align 8
    #dbg_declare(ptr %ptr.addr, !26, !DIExpression(), !27)
  call void @llvm.pseudoprobe(i64 5678655469166311522, i64 1, i32 0, i64 -1), !dbg !28
  %call = call noundef ptr @_Z6calleePv(ptr noundef null), !dbg !29
  ret ptr %call, !dbg !31
}

define dso_local noundef ptr @_Z3foov() #0 !dbg !32 {
entry:
  call void @llvm.pseudoprobe(i64 9191153033785521275, i64 1, i32 0, i64 -1), !dbg !35
  call void null(), !dbg !36
  call void @_Z3barv(), !dbg !38
  call void null(), !dbg !40
  %call = call noundef ptr @_Z3barPv(ptr noundef null), !dbg !42
  ret ptr %call, !dbg !44
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #1

attributes #0 = { "use-sample-profile" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6, !7, !8, !9}
!llvm.ident = !{!10}
!llvm.pseudo_probe_desc = !{!11, !12, !13, !14}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "test.cc", directory: "/tmp", checksumkind: CSK_MD5, checksum: "e16862f6a655f30cd332532a91f867b6")
!2 = !{!3}
!3 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 64)
!4 = !DISubroutineType(types: !5)
!5 = !{null}
!6 = !{i32 7, !"Dwarf Version", i32 5}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 7, !"uwtable", i32 2}
!9 = !{i32 7, !"frame-pointer", i32 2}
!10 = !{!"clang"}
!11 = !{i64 7108221232740920931, i64 4294967295, !"_Z6calleePv"}
!12 = !{i64 -1069303473483922844, i64 4294967295, !"_Z3barv"}
!13 = !{i64 5678655469166311522, i64 281479271677951, !"_Z3barPv"}
!14 = !{i64 9191153033785521275, i64 1125904201809919, !"_Z3foov"}
!15 = distinct !DISubprogram(name: "callee", linkageName: "_Z6calleePv", scope: !1, file: !1, line: 1, type: !16, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !19)
!16 = !DISubroutineType(types: !17)
!17 = !{!18, !18}
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!19 = !{}
!20 = !DILocalVariable(name: "ptr", arg: 1, scope: !15, file: !1, line: 1, type: !18)
!21 = !DILocation(line: 1, column: 20, scope: !15)
!22 = !DILocation(line: 2, column: 5, scope: !15)
!23 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !1, file: !1, line: 5, type: !4, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!24 = !DILocation(line: 5, column: 13, scope: !23)
!25 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barPv", scope: !1, file: !1, line: 7, type: !16, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !19)
!26 = !DILocalVariable(name: "ptr", arg: 1, scope: !25, file: !1, line: 7, type: !18)
!27 = !DILocation(line: 7, column: 17, scope: !25)
!28 = !DILocation(line: 8, column: 12, scope: !25)
!29 = !DILocation(line: 8, column: 12, scope: !30)
!30 = !DILexicalBlockFile(scope: !25, file: !1, discriminator: 455082007)
!31 = !DILocation(line: 8, column: 5, scope: !25)
!32 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 11, type: !33, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!33 = !DISubroutineType(types: !34)
!34 = !{!18}
!35 = !DILocation(line: 12, column: 5, scope: !32)
!36 = !DILocation(line: 12, column: 5, scope: !37)
!37 = !DILexicalBlockFile(scope: !32, file: !1, discriminator: 387973143)
!38 = !DILocation(line: 13, column: 5, scope: !39)
!39 = !DILexicalBlockFile(scope: !32, file: !1, discriminator: 455082015)
!40 = !DILocation(line: 14, column: 5, scope: !41)
!41 = !DILexicalBlockFile(scope: !32, file: !1, discriminator: 387973159)
!42 = !DILocation(line: 15, column: 12, scope: !43)
!43 = !DILexicalBlockFile(scope: !32, file: !1, discriminator: 455082031)
!44 = !DILocation(line: 15, column: 5, scope: !32)
