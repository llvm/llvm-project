; RUN: llc < %s -mtriple=arm64-windows -filetype=obj -aarch64-enable-atomic-cfg-tidy=0 | llvm-readobj --codeview - | FileCheck %s --check-prefix=OBJ

; Generated from:
; volatile int x;
; int getint(void);
; void putint(int);
; static inline int inlineinc(int a) {
;   int b = a + 1;
;   ++x;
;   return b;
; }
; void f(int p) {
;   if (p) {
;     int a = getint();
;     int b = inlineinc(a);
;     putint(b);
;   } else {
;     int c = getint();
;     putint(c);
;   }
; }


; OBJ:   DefRangeRegisterRelSym {
; OBJ:     Kind: S_DEFRANGE_REGISTER_REL (0x1145)
; OBJ:     BaseRegister: ARM64_SP (0x51)
; OBJ:     HasSpilledUDTMember: No
; OBJ:     OffsetInParent: 0
; OBJ:     BasePointerOffset: 12
; OBJ:     LocalVariableAddrRange {
; OBJ:       OffsetStart: .text+0x14
; OBJ:       ISectStart: 0x0
; OBJ:       Range: 0x30
; OBJ:     }
; OBJ:   }

; ModuleID = 't.cpp'
source_filename = "test/DebugInfo/COFF/register-variables-arm64.ll"
target datalayout = "e-m:w-p:64:64-i32:32-i64:64-i128:128-n32:64-S128"
target triple = "arm64-unknown-windows-msvc19.16.27023"

@x = common dso_local global i32 0, align 4, !dbg !0

define dso_local void @f(i32 %p) !dbg !11 {
entry:
  %p.addr = alloca i32, align 4
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %c = alloca i32, align 4
  store i32 %p, ptr %p.addr, align 4
    #dbg_declare(ptr %p.addr, !14, !DIExpression(), !15)
  %0 = load i32, ptr %p.addr, align 4, !dbg !16
  %tobool = icmp ne i32 %0, 0, !dbg !16
  %call = call i32 @getint(), !dbg !17
  br i1 %tobool, label %if.then, label %if.else, !dbg !16

if.then:                                          ; preds = %entry
    #dbg_declare(ptr %a, !19, !DIExpression(), !21)
  store i32 %call, ptr %a, align 4, !dbg !21
    #dbg_declare(ptr %b, !22, !DIExpression(), !23)
  %1 = load i32, ptr %a, align 4, !dbg !23
  %call1 = call i32 @inlineinc(i32 %1), !dbg !23
  store i32 %call1, ptr %b, align 4, !dbg !23
  %2 = load i32, ptr %b, align 4, !dbg !24
  br label %if.end, !dbg !25

if.else:                                          ; preds = %entry
    #dbg_declare(ptr %c, !26, !DIExpression(), !28)
  store i32 %call, ptr %c, align 4, !dbg !28
  %3 = load i32, ptr %c, align 4, !dbg !29
  br label %if.end, !dbg !30

if.end:                                           ; preds = %if.else, %if.then
  %.sink = phi i32 [ %3, %if.else ], [ %2, %if.then ]
  call void @putint(i32 %.sink), !dbg !17
  ret void, !dbg !31
}

declare dso_local i32 @getint()

define internal i32 @inlineinc(i32 %a) !dbg !32 {
entry:
  %a.addr = alloca i32, align 4
  %b = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
    #dbg_declare(ptr %a.addr, !35, !DIExpression(), !36)
    #dbg_declare(ptr %b, !37, !DIExpression(), !38)
  %0 = load i32, ptr %a.addr, align 4, !dbg !38
  %add = add nsw i32 %0, 1, !dbg !38
  store i32 %add, ptr %b, align 4, !dbg !38
  %1 = load volatile i32, ptr @x, align 4, !dbg !39
  %inc = add nsw i32 %1, 1, !dbg !39
  store volatile i32 %inc, ptr @x, align 4, !dbg !39
  %2 = load i32, ptr %b, align 4, !dbg !40
  ret i32 %2, !dbg !40
}

declare dso_local void @putint(i32)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8, !9, !10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 9.0.0 (trunk 361867) (llvm/trunk 361866)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "S:\\LLVM\\svn\\sbr\\bin", checksumkind: CSK_MD5, checksum: "734c448e95a6204a439a847ed063e5ce")
!4 = !{}
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !7)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{i32 2, !"CodeView", i32 1}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 2}
!11 = distinct !DISubprogram(name: "f", scope: !3, file: !3, line: 9, type: !12, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!12 = !DISubroutineType(types: !13)
!13 = !{null, !7}
!14 = !DILocalVariable(name: "p", arg: 1, scope: !11, file: !3, line: 9, type: !7)
!15 = !DILocation(line: 9, scope: !11)
!16 = !DILocation(line: 10, scope: !11)
!17 = !DILocation(line: 0, scope: !18)
!18 = distinct !DILexicalBlock(scope: !11, file: !3, line: 10)
!19 = !DILocalVariable(name: "a", scope: !20, file: !3, line: 11, type: !7)
!20 = distinct !DILexicalBlock(scope: !18, file: !3, line: 10)
!21 = !DILocation(line: 11, scope: !20)
!22 = !DILocalVariable(name: "b", scope: !20, file: !3, line: 12, type: !7)
!23 = !DILocation(line: 12, scope: !20)
!24 = !DILocation(line: 13, scope: !20)
!25 = !DILocation(line: 14, scope: !20)
!26 = !DILocalVariable(name: "c", scope: !27, file: !3, line: 15, type: !7)
!27 = distinct !DILexicalBlock(scope: !18, file: !3, line: 14)
!28 = !DILocation(line: 15, scope: !27)
!29 = !DILocation(line: 16, scope: !27)
!30 = !DILocation(line: 17, scope: !27)
!31 = !DILocation(line: 18, scope: !11)
!32 = distinct !DISubprogram(name: "inlineinc", scope: !3, file: !3, line: 4, type: !33, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !4)
!33 = !DISubroutineType(types: !34)
!34 = !{!7, !7}
!35 = !DILocalVariable(name: "a", arg: 1, scope: !32, file: !3, line: 4, type: !7)
!36 = !DILocation(line: 4, scope: !32)
!37 = !DILocalVariable(name: "b", scope: !32, file: !3, line: 5, type: !7)
!38 = !DILocation(line: 5, scope: !32)
!39 = !DILocation(line: 6, scope: !32)
!40 = !DILocation(line: 7, scope: !32)
