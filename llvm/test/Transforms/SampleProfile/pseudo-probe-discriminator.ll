; RUN: opt < %s -passes='default<O2>' -new-pm-debug-info-for-profiling -S | FileCheck %s --check-prefix=DEBUG
; RUN: opt < %s -passes='default<O2>' -new-pm-debug-info-for-profiling -new-pm-pseudo-probe-for-profiling -S | FileCheck %s --check-prefix=PROBE
; RUN: opt < %s -passes='thinlto-pre-link<O2>' -new-pm-debug-info-for-profiling -new-pm-pseudo-probe-for-profiling -S | FileCheck %s --check-prefix=PROBE


@a = dso_local global i32 0, align 4

; Function Attrs: uwtable
define void @_Z3foov(i32 %x) #0 !dbg !4 {
bb0:
  %cmp = icmp eq i32 %x, 0, !dbg !10
  br i1 %cmp, label %bb1, label %bb2

bb1:
; DEBUG:  call void @_Z3barv(), !dbg ![[CALL1:[0-9]+]]
; PROBE:  call void @_Z3barv(), !dbg ![[CALL1:[0-9]+]]
  call void @_Z3barv(), !dbg !10
; DEBUG:  call void @_Z3barv(), !dbg ![[CALL2:[0-9]+]]
; PROBE:  call void @_Z3barv(), !dbg ![[CALL2:[0-9]+]]
  call void @_Z3barv(), !dbg !11
  ret void, !dbg !13

bb2:
; DEBUG:  store i32 8, i32* @a, align 4, !dbg ![[INST:[0-9]+]]
; PROBE:  store i32 8, i32* @a, align 4, !dbg ![[INST:[0-9]+]]
  store i32 8, i32* @a, align 4, !dbg !12
  br label %bb3

bb3:
  ret void, !dbg !12
}

declare void @_Z3barv() #1
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) nounwind argmemonly
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) nounwind argmemonly

attributes #0 = { uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (trunk 250915) (llvm/trunk 251830)", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2)
!1 = !DIFile(filename: "c.cc", directory: "/tmp")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{!"clang version 3.8.0 (trunk 250915) (llvm/trunk 251830)"}
!10 = !DILocation(line: 4, column: 3, scope: !4)
!11 = !DILocation(line: 4, column: 9, scope: !4)
!12 = !DILocation(line: 4, column: 15, scope: !4)
!13 = !DILocation(line: 5, column: 1, scope: !4)

; DEBUG: ![[CALL1]] = !DILocation(line: 4, column: 3, scope: ![[CALL1BLOCK:[0-9]+]])
; DEBUG: ![[CALL1BLOCK]] = !DILexicalBlockFile({{.*}} discriminator: 2)
; DEBUG: ![[CALL2]] = !DILocation(line: 4, column: 9, scope: ![[CALL2BLOCK:[0-9]+]])
; DEBUG: ![[CALL2BLOCK]] = !DILexicalBlockFile({{.*}} discriminator: 8)
; DEBUG: ![[INST]] = !DILocation(line: 4, column: 15, scope: ![[INSTBLOCK:[0-9]+]])
; DEBUG: ![[INSTBLOCK]] = !DILexicalBlockFile({{.*}} discriminator: 4)

           
; PROBE: ![[CALL1]] = !DILocation(line: 4, column: 3, scope: ![[CALL1BLOCK:[0-9]+]])
; PROBE: ![[CALL1BLOCK]] = !DILexicalBlockFile({{.*}} discriminator: 186646575)
; PROBE: ![[CALL2]] = !DILocation(line: 4, column: 9, scope: ![[CALL2BLOCK:[0-9]+]])
; PROBE: ![[CALL2BLOCK]] = !DILexicalBlockFile({{.*}} discriminator: 186646583)
; PROBE: ![[INST]] = !DILocation(line: 4, column: 15, scope: ![[INSTBLOCK:[0-9]+]])
; PROBE: ![[INSTBLOCK]] = !DILexicalBlockFile({{.*}} discriminator: 4)
