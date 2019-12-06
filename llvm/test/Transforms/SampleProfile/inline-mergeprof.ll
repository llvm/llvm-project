; Test we lose details of not inlined profile without '-sample-profile-merge-inlinee'
; RUN: opt < %s -sample-profile -sample-profile-file=%S/Inputs/inline-mergeprof.prof -S | FileCheck -check-prefix=SCALE %s
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/inline-mergeprof.prof -S | FileCheck -check-prefix=SCALE %s

; Test we properly merge not inlined profile properly with '-sample-profile-merge-inlinee'
; RUN: opt < %s -sample-profile -sample-profile-file=%S/Inputs/inline-mergeprof.prof -sample-profile-merge-inlinee -S | FileCheck -check-prefix=MERGE %s
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/inline-mergeprof.prof -sample-profile-merge-inlinee -S | FileCheck -check-prefix=MERGE  %s

@.str = private unnamed_addr constant [11 x i8] c"sum is %d\0A\00", align 1

define i32 @main() !dbg !6 {
entry:
  %retval = alloca i32, align 4
  %s = alloca i32, align 4
  %i = alloca i32, align 4
  %tmp = load i32, i32* %i, align 4, !dbg !8
  %tmp1 = load i32, i32* %s, align 4, !dbg !8
  %call = call i32 @_Z3sumii(i32 %tmp, i32 %tmp1), !dbg !8
; SCALE: call i32 @_Z3sumii
; MERGE: call i32 @_Z3sumii
  store i32 %call, i32* %s, align 4, !dbg !8
  ret i32 0, !dbg !11
}

define i32 @_Z3sumii(i32 %x, i32 %y) !dbg !12 {
entry:
  %x.addr = alloca i32, align 4
  %y.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  store i32 %y, i32* %y.addr, align 4
  %tmp = load i32, i32* %x.addr, align 4, !dbg !13
  %tmp1 = load i32, i32* %y.addr, align 4, !dbg !13
  %add = add nsw i32 %tmp, %tmp1, !dbg !13
  %tmp2 = load i32, i32* %x.addr, align 4, !dbg !13
  %tmp3 = load i32, i32* %y.addr, align 4, !dbg !13
  %cmp1 = icmp ne i32 %tmp3, 100, !dbg !13
  br i1 %cmp1, label %if.then, label %if.else, !dbg !13

if.then:                                          ; preds = %entry
  %call = call i32 @_Z3subii(i32 %tmp2, i32 %tmp3), !dbg !14
  ret i32 %add, !dbg !14

if.else:                                          ; preds = %entry
  ret i32 %add, !dbg !15
}

define i32 @_Z3subii(i32 %x, i32 %y) !dbg !16 {
entry:
  %x.addr = alloca i32, align 4
  %y.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  store i32 %y, i32* %y.addr, align 4
  %tmp = load i32, i32* %x.addr, align 4, !dbg !17
  %tmp1 = load i32, i32* %y.addr, align 4, !dbg !17
  %add = sub nsw i32 %tmp, %tmp1, !dbg !17
  ret i32 %add, !dbg !18
}

declare i32 @printf(i8*, ...)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.5 ", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "calls.cc", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 1, !"Debug Info Version", i32 3}
!5 = !{!"clang version 3.5 "}
!6 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 7, type: !7, scopeLine: 7, virtualIndex: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 10, scope: !9)
!9 = !DILexicalBlockFile(scope: !10, file: !1, discriminator: 2)
!10 = distinct !DILexicalBlock(scope: !6, file: !1, line: 10)
!11 = !DILocation(line: 12, scope: !6)
!12 = distinct !DISubprogram(name: "sum", scope: !1, file: !1, line: 3, type: !7, scopeLine: 3, virtualIndex: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!13 = !DILocation(line: 4, scope: !12)
!14 = !DILocation(line: 5, scope: !12)
!15 = !DILocation(line: 6, scope: !12)
!16 = distinct !DISubprogram(name: "sub", scope: !1, file: !1, line: 20, type: !7, scopeLine: 20, virtualIndex: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!17 = !DILocation(line: 20, scope: !16)
!18 = !DILocation(line: 21, scope: !16)

; SCALE: name: "sum"
; SCALE-NEXT: {!"function_entry_count", i64 46}
; SCALE: !{!"branch_weights", i32 11, i32 2}
; SCALE: !{!"branch_weights", i64 20}
; SCALE: name: "sub"
; SCALE-NEXT: {!"function_entry_count", i64 -1}

; MERGE: name: "sum"
; MERGE-NEXT: {!"function_entry_count", i64 46}
; MERGE: !{!"branch_weights", i32 11, i32 23}
; MERGE: !{!"branch_weights", i32 10}
; MERGE: name: "sub"
; MERGE-NEXT: {!"function_entry_count", i64 3}