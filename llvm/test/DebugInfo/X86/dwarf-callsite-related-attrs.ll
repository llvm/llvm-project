; $ clang++ -S -emit-llvm -o - -gdwarf-5 -o - -O1 tail2.cc
; volatile int sink;
; void __attribute__((noinline)) bat() { sink++; }
; void __attribute__((noinline)) bar() { sink++; }
; void __attribute__((noinline)) foo() {
;   bar(); bat();
;   bar(); bat();
; }
; int __attribute__((disable_tail_calls)) main() { foo(); }

; On Windows, we don't handle the relocations needed for AT_return_pc properly
; and fail with "failed to compute relocation: IMAGE_REL_AMD64_ADDR32".
; UNSUPPORTED: cygwin,windows-gnu,windows-msvc

; RUN: %llc_dwarf -mtriple=x86_64-- < %s -o - | FileCheck %s -check-prefix=ASM
; RUN: %llc_dwarf -debugger-tune=lldb -mtriple=x86_64-- < %s -filetype=obj -o %t.o
; RUN: llvm-dwarfdump %t.o -o - | FileCheck %s -check-prefix=OBJ -implicit-check-not=DW_TAG_call -implicit-check-not=DW_AT_call
; RUN: llvm-dwarfdump -verify %t.o 2>&1 | FileCheck %s -check-prefix=VERIFY
; RUN: llvm-dwarfdump -statistics %t.o | FileCheck %s -check-prefix=STATS
; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis -o /dev/null

; VERIFY: No errors.
; STATS: "call site DIEs":6

@sink = global i32 0, align 4, !dbg !0

define void @__has_no_subprogram() {
entry:
  %0 = load volatile i32, i32* @sink, align 4
  %inc = add nsw i32 %0, 1
  store volatile i32 %inc, i32* @sink, align 4
  ret void
}

; ASM: DW_TAG_subprogram
; ASM:   DW_AT_call_all_calls
; OBJ: [[bat_sp:.*]]: DW_TAG_subprogram
; OBJ:   DW_AT_call_all_calls (true)
; OBJ:   DW_AT_name ("bat")
define void @_Z3batv() !dbg !13 {
entry:
  %0 = load volatile i32, i32* @sink, align 4, !dbg !16, !tbaa !17
  %inc = add nsw i32 %0, 1, !dbg !16
  store volatile i32 %inc, i32* @sink, align 4, !dbg !16, !tbaa !17
  ret void, !dbg !21
}

; ASM: DW_TAG_subprogram
; ASM:   DW_AT_call_all_calls
; OBJ: [[bar_sp:.*]]: DW_TAG_subprogram
; OBJ:   DW_AT_call_all_calls (true)
; OBJ:   DW_AT_name ("bar")
define void @_Z3barv() !dbg !22 {
entry:
  %0 = load volatile i32, i32* @sink, align 4, !dbg !23, !tbaa !17
  %inc = add nsw i32 %0, 1, !dbg !23
  store volatile i32 %inc, i32* @sink, align 4, !dbg !23, !tbaa !17
  ret void, !dbg !24
}

; ASM: DW_TAG_subprogram
; ASM:   DW_AT_call_all_calls
; OBJ: [[foo_sp:.*]]: DW_TAG_subprogram
; OBJ:   DW_AT_call_all_calls (true)
; OBJ:   DW_AT_name ("foo")
; OBJ:   DW_TAG_call_site
; OBJ:     DW_AT_call_origin ([[bar_sp]])
; OBJ:     DW_AT_call_return_pc
; OBJ:   DW_TAG_call_site
; OBJ:     DW_AT_call_origin ([[bat_sp]])
; OBJ:     DW_AT_call_return_pc
; OBJ:   DW_TAG_call_site
; OBJ:     DW_AT_call_origin ([[bar_sp]])
; OBJ:     DW_AT_call_return_pc
; OBJ:   DW_TAG_call_site
; OBJ:     DW_AT_call_origin ([[bat_sp]])
; OBJ:     DW_AT_call_tail_call
; OBJ:     DW_AT_call_pc
define void @_Z3foov() !dbg !25 {
entry:
  tail call void @__has_no_subprogram()
  tail call void @_Z3barv(), !dbg !26
  tail call void @_Z3batv(), !dbg !27
  tail call void @_Z3barv(), !dbg !26
  tail call void @_Z3batv(), !dbg !27
  ret void, !dbg !28
}

; ASM: DW_TAG_subprogram
; ASM: DW_AT_call_all_calls
; OBJ: DW_TAG_subprogram
; OBJ: DW_AT_call_all_calls (true)
; OBJ: DW_AT_name ("main")
; OBJ:   DW_TAG_call_site
; OBJ:     DW_AT_call_origin ([[foo_sp]])
; OBJ:     DW_AT_call_return_pc
; OBJ:   DW_TAG_call_site
; OBJ:     DW_AT_call_target
; OBJ:     DW_AT_call_return_pc
define i32 @main() !dbg !29 {
entry:
  call void @_Z3foov(), !dbg !32

  %indirect_target = load void ()*, void ()** undef
  call void %indirect_target()

  call void asm sideeffect "", "~{dirflag},~{fpsr},~{flags}"()

  ret i32 0, !dbg !33
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8, !9, !10, !11}
!llvm.ident = !{!12}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "sink", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 7.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "/Users/vsk/src/llvm.org-tailcall/tail2.cc", directory: "/Users/vsk/src/builds/llvm-project-tailcall-RA", checksumkind: CSK_MD5, checksum: "3b61952c21b7f657ddb7c0ad44cf5529")
!4 = !{}
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !7)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{i32 2, !"Dwarf Version", i32 5}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{i32 7, !"PIC Level", i32 2}
!12 = !{!"clang version 7.0.0 "}
!13 = distinct !DISubprogram(name: "bat", linkageName: "_Z3batv", scope: !3, file: !3, line: 2, type: !14, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, isOptimized: true, unit: !2, retainedNodes: !4)
!14 = !DISubroutineType(types: !15)
!15 = !{null}
!16 = !DILocation(line: 2, column: 44, scope: !13)
!17 = !{!18, !18, i64 0}
!18 = !{!"int", !19, i64 0}
!19 = !{!"omnipotent char", !20, i64 0}
!20 = !{!"Simple C++ TBAA"}
!21 = !DILocation(line: 2, column: 48, scope: !13)
!22 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !3, file: !3, line: 3, type: !14, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, isOptimized: true, unit: !2, retainedNodes: !4)
!23 = !DILocation(line: 3, column: 44, scope: !22)
!24 = !DILocation(line: 3, column: 48, scope: !22)
!25 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !3, file: !3, line: 4, type: !14, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, isOptimized: true, unit: !2, retainedNodes: !4)
!26 = !DILocation(line: 5, column: 3, scope: !25)
!27 = !DILocation(line: 6, column: 3, scope: !25)
!28 = !DILocation(line: 7, column: 1, scope: !25)
!29 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 8, type: !30, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, isOptimized: true, unit: !2, retainedNodes: !4)
!30 = !DISubroutineType(types: !31)
!31 = !{!7}
!32 = !DILocation(line: 8, column: 50, scope: !29)
!33 = !DILocation(line: 8, column: 57, scope: !29)
