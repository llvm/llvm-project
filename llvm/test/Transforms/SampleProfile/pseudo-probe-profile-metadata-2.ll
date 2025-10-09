; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/pseudo-probe-profile.prof -pass-remarks=sample-profile -S | FileCheck %s
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/pseudo-probe-profile.prof -pass-remarks=sample-profile -overwrite-existing-weights=1 -S | FileCheck %s -check-prefix=OVW

define dso_local i32 @foo(i32 %x, ptr %f) #0 !dbg !4 !prof !10 {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  %0 = load i32, ptr %x.addr, align 4
  %cmp = icmp eq i32 %0, 0
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 1, i32 0, i64 -1)
  br i1 %cmp, label %if.then, label %if.else, !prof !11

if.then:
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 2, i32 0, i64 -1)
  ; CHECK: call {{.*}}, !dbg ![[#]], !prof ![[#PROF:]]
  ; OVW: call {{.*}}, !dbg ![[#]], !prof ![[#PROF:]]
  call void %f(i32 1), !dbg !13, !prof !16
  store i32 1, ptr %retval, align 4
  br label %return

if.else:
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 2, i32 0, i64 0)
  ; CHECK: call {{.*}}, !dbg ![[#]], !prof  ![[#PROF]]
  ;; The block should have a 0 weight. Check the profile metadata is dropped.
  ; OVW-NOT: call {{.*}}, !dbg ![[#]], !prof
  call void %f(i32 2), !dbg !15, !prof !16
  store i32 2, ptr %retval, align 4
  br label %return

return:
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 6, i32 0, i64 -1)
  %1 = load i32, ptr %retval, align 4
  ret i32 %1
}

; CHECK: ![[#PROF]] = !{!"VP", i32 0, i64 7, i64 9191153033785521275, i64 5, i64 -1069303473483922844, i64 2}
; OVW: ![[#PROF]] = !{!"VP", i32 0, i64 7, i64 9191153033785521275, i64 5, i64 -1069303473483922844, i64 2}

declare void @llvm.pseudoprobe(i64, i64, i32, i64) #0

attributes #0 = {"use-sample-profile"}

!llvm.module.flags = !{!0, !1}
!llvm.pseudo_probe_desc = !{!2}

!0 = !{i32 7, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !{i64 6699318081062747564, i64 563022570642068, !"foo", null}
!4 = distinct !DISubprogram(name: "foo", scope: !5, file: !5, line: 9, type: !6, scopeLine: 9, spFlags: DISPFlagDefinition, unit: !9)
!5 = !DIFile(filename: "test.cpp", directory: "test")
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !8}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !5, isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug)
!10 = !{!"function_entry_count", i64 14}
!11 = !{!"branch_weights", i32 100, i32 0}
;; A discriminator of 186646559 which is 0xB20001F in hexdecimal, stands for an indirect call probe
;; with an index of 3 and probe factor of 1.0.
!12 = !DILexicalBlockFile(scope: !4, file: !5, discriminator: 186646559)
!13 = distinct !DILocation(line: 10, column: 11, scope: !12)
;; A discriminator of 134217759 which is 0x800001F in hexdecimal, stands for an indirect call probe
;; with an index of 3 and probe factor of 0.
!14 = !DILexicalBlockFile(scope: !4, file: !5, discriminator: 134217759)
!15 = distinct !DILocation(line: 10, column: 11, scope: !14)
!16 = !{!"VP", i32 0, i64 7, i64 9191153033785521275, i64 5, i64 -1069303473483922844, i64 2}
