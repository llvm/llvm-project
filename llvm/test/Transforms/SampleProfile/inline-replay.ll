;; Note that this needs new pass manager for now. Passing `-sample-profile-inline-replay` to legacy pass manager is a no-op.

;; Check baseline inline decisions
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/inline-topdown.prof -sample-profile-merge-inlinee -sample-profile-top-down-load -pass-remarks=inline -S 2>&1 | FileCheck -check-prefix=DEFAULT %s

;; Check replay inline decisions
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/inline-topdown.prof -sample-profile-inline-replay=%S/Inputs/inline-replay.txt -sample-profile-inline-replay-scope=Module  -sample-profile-merge-inlinee -sample-profile-top-down-load -pass-remarks=inline -S 2>&1 | FileCheck -check-prefix=REPLAY %s

;; Check baseline inline decisions with "inline-topdown-inline-all.prof" which inlines all sites
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/inline-topdown-inline-all.prof -sample-profile-merge-inlinee -sample-profile-top-down-load -pass-remarks=inline -S 2>&1 | FileCheck -check-prefix=DEFAULT-ALL %s

;; Check function scope replay inline decisions with "inline-topdown-inline-all.prof" and "inline-topdown-function-scope.txt" which only contains: '_Z3sumii' inlined into 'main'
;; 1. _Z3sumii is inlined into main, but all other inline candidates in main (e.g. _Z3subii) are not inlined
;; 2. Inline decisions made in other functions match default sample inlining, in this case _Z3subii is inlined into _Z3sumii
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/inline-topdown-inline-all.prof -sample-profile-inline-replay=%S/Inputs/inline-replay-function-scope.txt -sample-profile-inline-replay-scope=Function -sample-profile-merge-inlinee -sample-profile-top-down-load -pass-remarks=inline -S 2>&1 | FileCheck -check-prefix=REPLAY-ALL-FUNCTION %s

;; Check behavior on non-existent replay file
; RUN: not opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/inline-topdown.prof -sample-profile-inline-replay=%S -sample-profile-merge-inlinee -sample-profile-top-down-load -pass-remarks=inline -S 2>&1 | FileCheck -check-prefix=REPLAY-ERROR %s

;; Check scope inlining errors out on non <Module|Function> inputs
; RUN: not opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/inline-topdown.prof -sample-profile-inline-replay=%S/Inputs/inline-replay.txt -sample-profile-inline-replay-scope=function -sample-profile-merge-inlinee -sample-profile-top-down-load -pass-remarks=inline -S 2>&1 | FileCheck -check-prefix=REPLAY-ERROR-SCOPE %s

; DEFAULT: '_Z3sumii' inlined into 'main' to match profiling context with (cost={{[-0-9]+}}
; DEFAULT: '_Z3subii' inlined into '_Z3sumii' to match profiling context with (cost={{[-0-9]+}}
; DEFAULT-NOT: '_Z3subii' inlined into 'main'

; REPLAY: '_Z3sumii' inlined into 'main' to match profiling context with (cost=always)
; REPLAY: '_Z3subii' inlined into 'main' to match profiling context with (cost=always)
; REPLAY-NOT: '_Z3subii' inlined into '_Z3sumii'

; DEFAULT-ALL: '_Z3sumii' inlined into 'main' to match profiling context with (cost={{[-0-9]+}}
; DEFAULT-ALL: '_Z3subii' inlined into 'main' to match profiling context with (cost={{[-0-9]+}}
; DEFAULT-ALL: '_Z3subii' inlined into '_Z3sumii' to match profiling context with (cost={{[-0-9]+}}

; REPLAY-ALL-FUNCTION : _Z3sumii' inlined into 'main' to match profiling context with (cost=always)
; REPLAY-ALL-FUNCTION-NOT: '_Z3subii' inlined into 'main' to match profiling context with (cost={{[-0-9]+}}
; REPLAY-ALL-FUNCTION: '_Z3subii' inlined into '_Z3sumii' to match profiling context with (cost={{[-0-9]+}}

; REPLAY-ERROR: error: Could not open remarks file:
; REPLAY-ERROR-SCOPE: for the --sample-profile-inline-replay-scope option: Cannot find option named 'function'!

@.str = private unnamed_addr constant [11 x i8] c"sum is %d\0A\00", align 1

define i32 @_Z3sumii(i32 %x, i32 %y) #0 !dbg !6 {
entry:
  %x.addr = alloca i32, align 4
  %y.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  store i32 %y, i32* %y.addr, align 4
  %tmp = load i32, i32* %x.addr, align 4, !dbg !8
  %tmp1 = load i32, i32* %y.addr, align 4, !dbg !8
  %add = add nsw i32 %tmp, %tmp1, !dbg !8
  %tmp2 = load i32, i32* %x.addr, align 4, !dbg !8
  %tmp3 = load i32, i32* %y.addr, align 4, !dbg !8
  %call = call i32 @_Z3subii(i32 %tmp2, i32 %tmp3), !dbg !8
  ret i32 %add, !dbg !8
}

define i32 @_Z3subii(i32 %x, i32 %y) #0 !dbg !9 {
entry:
  %x.addr = alloca i32, align 4
  %y.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  store i32 %y, i32* %y.addr, align 4
  %tmp = load i32, i32* %x.addr, align 4, !dbg !10
  %tmp1 = load i32, i32* %y.addr, align 4, !dbg !10
  %add = sub nsw i32 %tmp, %tmp1, !dbg !10
  ret i32 %add, !dbg !11
}

define i32 @main() #0 !dbg !12 {
entry:
  %retval = alloca i32, align 4
  %s = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 0, i32* %retval
  store i32 0, i32* %i, align 4, !dbg !13
  br label %while.cond, !dbg !14

while.cond:                                       ; preds = %if.end, %entry
  %tmp = load i32, i32* %i, align 4, !dbg !15
  %inc = add nsw i32 %tmp, 1, !dbg !15
  store i32 %inc, i32* %i, align 4, !dbg !15
  %cmp = icmp slt i32 %tmp, 400000000, !dbg !15
  br i1 %cmp, label %while.body, label %while.end, !dbg !15

while.body:                                       ; preds = %while.cond
  %tmp1 = load i32, i32* %i, align 4, !dbg !17
  %cmp1 = icmp ne i32 %tmp1, 100, !dbg !17
  br i1 %cmp1, label %if.then, label %if.else, !dbg !17

if.then:                                          ; preds = %while.body
  %tmp2 = load i32, i32* %i, align 4, !dbg !19
  %tmp3 = load i32, i32* %s, align 4, !dbg !19
  %call = call i32 @_Z3sumii(i32 %tmp2, i32 %tmp3), !dbg !19
  store i32 %call, i32* %s, align 4, !dbg !19
  br label %if.end, !dbg !19

if.else:                                          ; preds = %while.body
  store i32 30, i32* %s, align 4, !dbg !21
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  br label %while.cond, !dbg !23

while.end:                                        ; preds = %while.cond
  %tmp4 = load i32, i32* %s, align 4, !dbg !25
  %call2 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i32 0, i32 0), i32 %tmp4), !dbg !25
  ret i32 0, !dbg !26
}

declare i32 @printf(i8*, ...)

attributes #0 = { "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.5 ", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "calls.cc", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 1, !"Debug Info Version", i32 3}
!5 = !{!"clang version 3.5 "}
!6 = distinct !DISubprogram(name: "sum", linkageName: "_Z3sumii", scope: !1, file: !1, line: 3, type: !7, scopeLine: 3, virtualIndex: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 4, scope: !6)
!9 = distinct !DISubprogram(name: "sub", linkageName: "_Z3subii", scope: !1, file: !1, line: 20, type: !7, scopeLine: 20, virtualIndex: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DILocation(line: 20, scope: !9)
!11 = !DILocation(line: 21, scope: !9)
!12 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 7, type: !7, scopeLine: 7, virtualIndex: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!13 = !DILocation(line: 8, scope: !12)
!14 = !DILocation(line: 9, scope: !12)
!15 = !DILocation(line: 9, scope: !16)
!16 = !DILexicalBlockFile(scope: !12, file: !1, discriminator: 2)
!17 = !DILocation(line: 10, scope: !18)
!18 = distinct !DILexicalBlock(scope: !12, file: !1, line: 10)
!19 = !DILocation(line: 10, scope: !20)
!20 = !DILexicalBlockFile(scope: !18, file: !1, discriminator: 2)
!21 = !DILocation(line: 10, scope: !22)
!22 = !DILexicalBlockFile(scope: !18, file: !1, discriminator: 4)
!23 = !DILocation(line: 10, scope: !24)
!24 = !DILexicalBlockFile(scope: !18, file: !1, discriminator: 6)
!25 = !DILocation(line: 11, scope: !12)
!26 = !DILocation(line: 12, scope: !12)
