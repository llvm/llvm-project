; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/coro-presplit-massage-I.prof -sample-profile-coro-massage -S | FileCheck %s
; RUN: llvm-profdata merge --sample --extbinary %S/Inputs/coro-presplit-massage-I.prof -o %t.extbinary.afdo
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%t.extbinary.afdo -sample-profile-coro-massage -S | FileCheck %s
; RUN: llvm-profdata merge --sample --extbinary --use-md5 %S/Inputs/coro-presplit-massage-I.prof -o %t.md5.afdo
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%t.md5.afdo -sample-profile-coro-massage -S | FileCheck %s
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/coro-presplit-massage-I.prof -S | FileCheck %s --check-prefix=OFF

; Test 1: Function entry count comes from .ramp (100 + 1 = 101) and ignores
; .resume's head samples (1000) and (0, 0) dispatch samples (1000). Clearing
; (0, 0) also prevents coro.cleanup from polluting init.suspend.
; CHECK-LABEL: define ptr @_Z15coro_entry_onlyv()
; CHECK-SAME: !prof ![[ENTRY_PROF:[0-9]+]]
; CHECK: br i1 %r0, label %init.ready, label %init.suspend, {{.*}}!prof ![[INIT_BR:[0-9]+]]

; CHECK-DAG: ![[ENTRY_PROF]] = !{!"function_entry_count", i64 101}
; CHECK-DAG: ![[INIT_BR]] = !{!"branch_weights", i32 102, i32 0}

; Without -sample-profile-coro-massage, the (0, 0) samples (100) land on
; coro.cleanup and flow into init.suspend. This inverts the branch to {2, 101}.
; OFF-LABEL: define ptr @_Z15coro_entry_onlyv()
; OFF: br i1 %r0, label %init.ready, label %init.suspend, {{.*}}!prof ![[OFF_INIT_BR:[0-9]+]]
; OFF: ![[OFF_INIT_BR]] = !{!"branch_weights", i32 2, i32 101}

declare void @_Z9coro_freePv(ptr)
declare i1 @_Z11init_ready_P16struct_awaitable(ptr)
declare void @_Z11init_resumeP16struct_awaitable(ptr)
declare void @_Z15coro_entry_onlyv.__await_suspend_wrapper__init(ptr, ptr) #1

define ptr @_Z15coro_entry_onlyv() #0 !dbg !6 {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr @_Z15coro_entry_onlyv, ptr null), !dbg !7
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null), !dbg !7
  %r0 = call i1 @_Z11init_ready_P16struct_awaitable(ptr null), !dbg !8
  br i1 %r0, label %init.ready, label %init.suspend, !dbg !8

init.suspend:
  %s0 = call token @llvm.coro.save(ptr null), !dbg !7
  call void @llvm.coro.await.suspend.void(ptr null, ptr %hdl, ptr @_Z15coro_entry_onlyv.__await_suspend_wrapper__init), !dbg !7
  %sp0 = call i8 @llvm.coro.suspend(token %s0, i1 false), !dbg !7
  switch i8 %sp0, label %coro.ret [
    i8 0, label %init.ready
    i8 1, label %coro.cleanup
  ], !dbg !9

init.ready:
  call void @_Z11init_resumeP16struct_awaitable(ptr null), !dbg !10
  br label %coro.ret, !dbg !10

coro.cleanup:
  %free_ptr = call ptr @llvm.coro.free(token %id, ptr %hdl), !dbg !7
  call void @_Z9coro_freePv(ptr %free_ptr), !dbg !7
  br label %coro.ret, !dbg !7

coro.ret:
  call void @llvm.coro.end(ptr null, i1 false, token none), !dbg !7
  ret ptr %hdl, !dbg !7
}

declare token @llvm.coro.id(i32, ptr readnone, ptr readonly, ptr)
declare ptr @llvm.coro.begin(token, ptr writeonly)
declare token @llvm.coro.save(ptr)
declare void @llvm.coro.await.suspend.void(ptr, ptr, ptr)
declare i8 @llvm.coro.suspend(token, i1)
declare ptr @llvm.coro.free(token, ptr readonly)
declare void @llvm.coro.end(ptr, i1, token)

attributes #0 = { presplitcoroutine "use-sample-profile" }
attributes #1 = { "sample-profile-suffix-elision-policy"="selected" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, debugInfoForProfiling: true)
!1 = !DIFile(filename: "coro.cpp", directory: ".")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !DISubroutineType(types: !2)
!6 = distinct !DISubprogram(name: "coro_entry_only", linkageName: "_Z15coro_entry_onlyv", scope: !1, file: !1, line: 10, type: !5, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!7 = !DILocation(line: 10, column: 1, scope: !6)
!8 = !DILocation(line: 10, column: 5, scope: !11)
!9 = !DILocation(line: 10, column: 5, scope: !12)
!10 = !DILocation(line: 10, column: 9, scope: !13)
!11 = !DILexicalBlockFile(scope: !6, file: !1, discriminator: 4)
!12 = !DILexicalBlockFile(scope: !6, file: !1, discriminator: 2)
!13 = !DILexicalBlockFile(scope: !6, file: !1, discriminator: 12)
