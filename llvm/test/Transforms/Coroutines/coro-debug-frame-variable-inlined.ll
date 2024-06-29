; RUN: opt < %s -passes='module(coro-early),cgscc(inline,coro-split)' -S | FileCheck %s
; RUN: opt --try-experimental-debuginfo-iterators < %s -passes='module(coro-early),cgscc(inline,coro-split)' -S | FileCheck %s

; Simplified version from pr#75104.
; Make sure we do not update debug location for hosited dbg.declare intrinsics when optimizing coro frame.

; CHECK-NOT: mismatched subprogram between llvm.dbg.declare variable and !dbg attachment

%"struct.std::coroutine_handle" = type { i8 }

define void @_Z1fv() presplitcoroutine {
entry:
  %0 = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %1 = call ptr @llvm.coro.begin(token %0, ptr null), !dbg !10
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %entry
  call void @_ZN1BD1Ev()
  %2 = call token @llvm.coro.save(ptr null)
  %3 = call i8 @llvm.coro.suspend(token none, i1 false)
  br label %for.cond
}

declare token @llvm.coro.id(i32, ptr readnone, ptr nocapture readonly, ptr)
declare ptr @llvm.coro.begin(token, ptr writeonly)
declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare token @llvm.coro.save(ptr)
declare i8 @llvm.coro.suspend(token, i1)

define void @_ZN1BD1Ev() {
entry:
  %b11 = alloca [0 x [0 x %"struct.std::coroutine_handle"]], i32 0, align 1
  call void @llvm.dbg.declare(metadata ptr %b11, metadata !13, metadata !DIExpression()), !dbg !21
  %call = call i1 @_ZNSt16coroutine_handleIvEcvbEv(ptr %b11), !dbg !21
  ret void
}

declare i1 @_ZNSt16coroutine_handleIvEcvbEv(ptr)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 18.0.0git", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "")
!2 = !{!3}
!3 = !DIGlobalVariableExpression(var: !4, expr: !DIExpression())
!4 = distinct !DIGlobalVariable(name: "a", scope: !0, file: !5, line: 17, type: !6, isLocal: false, isDefinition: true)
!5 = !DIFile(filename: "bad.cpp", directory: "")
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "coroutine_handle<void>", scope: !7, file: !5, line: 2, size: 8, flags: DIFlagTypePassByValue, elements: !8, templateParams: !8, identifier: "_ZTSSt16coroutine_handleIvE")
!7 = !DINamespace(name: "std", scope: null)
!8 = !{}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !DILocation(line: 31, column: 7, scope: !11)
!11 = distinct !DISubprogram(name: "f", linkageName: "_Z1fv", scope: !5, file: !5, line: 31, type: !12, scopeLine: 31, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!12 = distinct !DISubroutineType(types: !8)
!13 = !DILocalVariable(name: "b", scope: !14, file: !5, line: 27, type: !6)
!14 = distinct !DILexicalBlock(scope: !15, file: !5, line: 27, column: 14)
!15 = distinct !DILexicalBlock(scope: !16, file: !5, line: 26, column: 8)
!16 = distinct !DISubprogram(name: "~B", linkageName: "_ZN1BD2Ev", scope: !17, file: !5, line: 26, type: !18, scopeLine: 26, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, declaration: !20, retainedNodes: !8)
!17 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "B", file: !5, line: 18, size: 8, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !8, identifier: "_ZTS1B")
!18 = !DISubroutineType(types: !19)
!19 = !{null}
!20 = !DISubprogram(name: "~B", scope: !17, file: !5, line: 26, type: !18, scopeLine: 26, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!21 = !DILocation(line: 27, column: 14, scope: !14)
