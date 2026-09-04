; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s --check-prefix=SIMPLE
; RUN: llc < %s | llvm-mc -filetype=obj --triple=x86_64-windows | llvm-readobj - --codeview | FileCheck %s --check-prefix=SIMPLE
; RUN: llc < %s -O0 -enable-tail-merge=0 | FileCheck %s --check-prefix=MULTI-ASM
; RUN: llc < %s -O0 -enable-tail-merge=0 -filetype=obj | llvm-readobj - --codeview | FileCheck %s --check-prefix=MULTI
; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s --check-prefix=NOEPILOGUE

; Check that the compatibility S_REGREL32 record used for an implicit C++
; `this` pointer is bounded by the containing procedure's debug range,
; including when the function has multiple epilogues or no epilogue.

; MULTI-ASM-LABEL: "?f@foo@@QEAAXH@Z":
; MULTI-ASM-COUNT-3: retq

; SIMPLE:      GlobalProcIdSym {
; SIMPLE:        CodeSize: 0x{{[1-9A-Fa-f][0-9A-Fa-f]*}}
; SIMPLE:        DbgStart: 0x{{[1-9A-Fa-f][0-9A-Fa-f]*}}
; SIMPLE:        DbgEnd: 0x{{[1-9A-Fa-f][0-9A-Fa-f]*}}
; SIMPLE:        DisplayName: foo::foo
; SIMPLE:      }
; SIMPLE:      RegRelativeSym {
; SIMPLE-NEXT:   Kind: S_REGREL32 (0x1111)
; SIMPLE:        VarName: this
; SIMPLE:      }

; MULTI:      GlobalProcIdSym {
; MULTI:        CodeSize: 0x{{[1-9A-Fa-f][0-9A-Fa-f]*}}
; MULTI:        DbgStart: 0x{{[1-9A-Fa-f][0-9A-Fa-f]*}}
; MULTI:        DbgEnd: 0x{{[1-9A-Fa-f][0-9A-Fa-f]*}}
; MULTI-LABEL: DisplayName: foo::f
; MULTI:        LinkageName: ?f@foo@@QEAAXH@Z
; MULTI-COUNT-1: RegRelativeSym {
; MULTI-NEXT:   Kind: S_REGREL32 (0x1111)
; MULTI:        VarName: this

; NOEPILOGUE-LABEL: {{^ *ProcIdSym \{$}}
; NOEPILOGUE:        CodeSize: 0x[[NOEPILOGUE_SIZE:[1-9A-Fa-f][0-9A-Fa-f]*]]
; NOEPILOGUE:        DbgStart: 0x{{[1-9A-Fa-f][0-9A-Fa-f]*}}
; NOEPILOGUE:        DbgEnd: 0x[[NOEPILOGUE_SIZE]]
; NOEPILOGUE:        DisplayName: foo::g
; NOEPILOGUE:        RegRelativeSym {
; NOEPILOGUE-NEXT:   Kind: S_REGREL32 (0x1111)
; NOEPILOGUE:        VarName: this

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.11.25507"

%class.foo = type { i32, i32 }

$"\01??0foo@@QEAA@XZ" = comdat any
$"?f@foo@@QEAAXH@Z" = comdat any

; Function Attrs: noinline optnone uwtable
define linkonce_odr void @"\01??0foo@@QEAA@XZ"(ptr %this) #0 comdat align 2 !dbg !10 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %this.addr, metadata !11, metadata !DIExpression()), !dbg !12
  %this1 = load ptr, ptr %this.addr, align 8
  %a = getelementptr inbounds %class.foo, ptr %this1, i32 0, i32 0
  store i32 1, ptr %a, align 4, !dbg !13
  %b = getelementptr inbounds %class.foo, ptr %this1, i32 0, i32 1
  store i32 2, ptr %b, align 4, !dbg !13
  ret void, !dbg !13
}

declare void @exit_a(ptr)
declare void @exit_b(ptr)
declare void @exit_c(ptr)
declare void @throw_now(ptr) #1

; Function Attrs: noinline optnone uwtable
define linkonce_odr void @"?f@foo@@QEAAXH@Z"(ptr %this, i32 %value) #0 comdat align 2 !dbg !22 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %this.addr, metadata !23,
                               metadata !DIExpression()), !dbg !24
  %negative = icmp slt i32 %value, 0, !dbg !25
  br i1 %negative, label %exit.a, label %check.zero, !dbg !25

exit.a:
  %this.a = load ptr, ptr %this.addr, align 8, !dbg !26
  call void @exit_a(ptr %this.a), !dbg !26
  ret void, !dbg !26

check.zero:
  %zero = icmp eq i32 %value, 0, !dbg !27
  br i1 %zero, label %exit.b, label %exit.c, !dbg !27

exit.b:
  %this.b = load ptr, ptr %this.addr, align 8, !dbg !28
  call void @exit_b(ptr %this.b), !dbg !28
  ret void, !dbg !28

exit.c:
  %this.c = load ptr, ptr %this.addr, align 8, !dbg !29
  call void @exit_c(ptr %this.c), !dbg !29
  ret void, !dbg !29
}

; Function Attrs: noinline noreturn optnone uwtable
define internal void @"?g@foo@@QEAAXXZ"(ptr %this) #0 align 2 !dbg !32 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %this.addr, metadata !31,
                               metadata !DIExpression()), !dbg !33
  call void @throw_now(ptr %this), !dbg !34
  unreachable
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

attributes #0 = { noinline optnone uwtable "frame-pointer"="none" }
attributes #1 = { noreturn nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!14, !15, !16}
!llvm.ident = !{!17}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1,
  producer: "clang", isOptimized: false, runtimeVersion: 0,
  emissionKind: FullDebug, enums: !2, retainedTypes: !2)
!1 = !DIFile(filename: "test.cpp", directory: "C:\\src")
!2 = !{}
!3 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "foo", file: !1,
  line: 1, size: 64, align: 64, elements: !4, identifier: ".?AVfoo@@")
!4 = !{!5}
!5 = !DISubprogram(name: "foo", linkageName: "\01??0foo@@QEAA@XZ", scope: !3,
  file: !1, line: 3, type: !6, isLocal: false, isDefinition: false,
  scopeLine: 3, flags: DIFlagPublic | DIFlagPrototyped)
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8}
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64,
  align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64,
  align: 64)
!10 = distinct !DISubprogram(name: "foo", linkageName: "\01??0foo@@QEAA@XZ",
  scope: !3, file: !1, line: 3, type: !6, isLocal: false,
  scopeLine: 3, flags: DIFlagPublic | DIFlagPrototyped,
  spFlags: DISPFlagDefinition, unit: !0,
  declaration: !5, retainedNodes: !2)
!11 = !DILocalVariable(name: "this", arg: 1, scope: !10, type: !9,
  flags: DIFlagArtificial | DIFlagObjectPointer)
!12 = !DILocation(line: 0, scope: !10)
!13 = !DILocation(line: 3, column: 17, scope: !10)
!14 = !{i32 2, !"CodeView", i32 1}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{i32 1, !"wchar_size", i32 2}
!17 = !{!"clang"}
!18 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!19 = !DISubroutineType(types: !20)
!20 = !{null, !9, !18}
!21 = !DISubprogram(name: "f", linkageName: "?f@foo@@QEAAXH@Z",
  scope: !3, file: !1, line: 5, type: !19, isLocal: false,
  isDefinition: false, scopeLine: 5,
  flags: DIFlagPublic | DIFlagPrototyped)
!22 = distinct !DISubprogram(name: "f", linkageName: "?f@foo@@QEAAXH@Z",
  scope: !3, file: !1, line: 5, type: !19, isLocal: false,
  scopeLine: 5, flags: DIFlagPublic | DIFlagPrototyped,
  spFlags: DISPFlagDefinition, unit: !0, declaration: !21,
  retainedNodes: !2)
!23 = !DILocalVariable(name: "this", arg: 1, scope: !22, type: !9,
  flags: DIFlagArtificial | DIFlagObjectPointer)
!24 = !DILocation(line: 0, scope: !22)
!25 = !DILocation(line: 6, column: 5, scope: !22)
!26 = !DILocation(line: 7, column: 5, scope: !22)
!27 = !DILocation(line: 8, column: 5, scope: !22)
!28 = !DILocation(line: 9, column: 5, scope: !22)
!29 = !DILocation(line: 10, column: 5, scope: !22)
!30 = !DISubprogram(name: "g", linkageName: "?g@foo@@QEAAXXZ", scope: !3,
  file: !1, line: 13, type: !6, isLocal: false, isDefinition: false,
  scopeLine: 13,
  flags: DIFlagPublic | DIFlagPrototyped)
!31 = !DILocalVariable(name: "this", arg: 1, scope: !32, type: !9,
  flags: DIFlagArtificial | DIFlagObjectPointer)
!32 = distinct !DISubprogram(name: "g", linkageName: "?g@foo@@QEAAXXZ",
  scope: !3, file: !1, line: 13, type: !6, isLocal: false, scopeLine: 13,
  flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDefinition,
  unit: !0, declaration: !30, retainedNodes: !2)
!33 = !DILocation(line: 0, scope: !32)
!34 = !DILocation(line: 14, column: 5, scope: !32)
