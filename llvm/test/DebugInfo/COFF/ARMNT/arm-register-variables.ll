; RUN: llc < %s -filetype=obj -arm-atomic-cfg-tidy=0 | llvm-readobj --codeview - | FileCheck %s --check-prefix=OBJ

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

;      OBJ:   Compile3Sym {
; OBJ-NEXT:     Kind: S_COMPILE3 (0x113C)
; OBJ-NEXT:     Language: C (0x0)
; OBJ-NEXT:     Flags [ (0x4000)
; OBJ-NEXT:       HotPatch (0x4000)
; OBJ-NEXT:     ]
; OBJ-NEXT:     Machine: ARMNT (0xF4)

;      OBJ: LocalSym {
; OBJ-NEXT:   Kind: S_LOCAL (0x113E)
; OBJ-NEXT:   Type: int (0x74)
; OBJ-NEXT:   Flags [ (0x1)
; OBJ-NEXT:     IsParameter (0x1)
; OBJ-NEXT:   ]
; OBJ-NEXT:   VarName: p
; OBJ-NEXT: }
; OBJ-NEXT: DefRangeRegisterRelSym {
; OBJ-NEXT:   Kind: S_DEFRANGE_REGISTER_REL (0x1145)
; OBJ-NEXT:   BaseRegister: ARM_SP (0x17)
; OBJ-NEXT:   HasSpilledUDTMember: No
; OBJ-NEXT:   OffsetInParent: 0
; OBJ-NEXT:   BasePointerOffset: 12
; OBJ-NEXT:   LocalVariableAddrRange {
; OBJ-NEXT:     OffsetStart: .text+0x8
; OBJ-NEXT:     ISectStart: 0x0
; OBJ-NEXT:     Range: 0x1A
; OBJ-NEXT:   }
; OBJ-NEXT: }

; ModuleID = 'test.c'
source_filename = "test.c"
target datalayout = "e-m:w-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7-pc-windows-msvc19.11.0"

@x = dso_local global i32 0, align 4, !dbg !0

define dso_local arm_aapcs_vfpcc void @f(i32 %p) !dbg !14 {
entry:
  %p.addr = alloca i32, align 4
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %c = alloca i32, align 4
  store i32 %p, ptr %p.addr, align 4
    #dbg_declare(ptr %p.addr, !17, !DIExpression(), !18)
  %0 = load i32, ptr %p.addr, align 4, !dbg !19
  %tobool = icmp ne i32 %0, 0, !dbg !19
  %call = call arm_aapcs_vfpcc i32 @getint(), !dbg !20
  br i1 %tobool, label %if.then, label %if.else, !dbg !19

if.then:                                          ; preds = %entry
    #dbg_declare(ptr %a, !22, !DIExpression(), !24)
  store i32 %call, ptr %a, align 4, !dbg !24
    #dbg_declare(ptr %b, !25, !DIExpression(), !26)
  %1 = load i32, ptr %a, align 4, !dbg !26
  %call1 = call arm_aapcs_vfpcc i32 @inlineinc(i32 %1), !dbg !26
  store i32 %call1, ptr %b, align 4, !dbg !26
  %2 = load i32, ptr %b, align 4, !dbg !27
  br label %if.end, !dbg !28

if.else:                                          ; preds = %entry
    #dbg_declare(ptr %c, !29, !DIExpression(), !31)
  store i32 %call, ptr %c, align 4, !dbg !31
  %3 = load i32, ptr %c, align 4, !dbg !32
  br label %if.end, !dbg !33

if.end:                                           ; preds = %if.else, %if.then
  %.sink = phi i32 [ %3, %if.else ], [ %2, %if.then ]
  call arm_aapcs_vfpcc void @putint(i32 %.sink), !dbg !20
  ret void, !dbg !34
}

declare dso_local arm_aapcs_vfpcc i32 @getint()

define internal arm_aapcs_vfpcc i32 @inlineinc(i32 %a) !dbg !35 {
entry:
  %a.addr = alloca i32, align 4
  %b = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
    #dbg_declare(ptr %a.addr, !38, !DIExpression(), !39)
    #dbg_declare(ptr %b, !40, !DIExpression(), !41)
  %0 = load i32, ptr %a.addr, align 4, !dbg !41
  %add = add nsw i32 %0, 1, !dbg !41
  store i32 %add, ptr %b, align 4, !dbg !41
  %1 = load volatile i32, ptr @x, align 4, !dbg !42
  %inc = add nsw i32 %1, 1, !dbg !42
  store volatile i32 %inc, ptr @x, align 4, !dbg !42
  %2 = load i32, ptr %b, align 4, !dbg !43
  ret i32 %2, !dbg !43
}

declare dso_local arm_aapcs_vfpcc void @putint(i32)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!9, !10, !11, !12}
!llvm.ident = !{!13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !6, line: 1, type: !7, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 12.0.0 (https://github.com/llvm/llvm-project.git fc031d29bea856f2b91a250fd81c5f9fb79dbe07)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "F:\\tmp\\test.c", directory: "F:\\tmp", checksumkind: CSK_MD5, checksum: "5fbd15e58dd6931fc3081de308d52889")
!4 = !{}
!5 = !{!0}
!6 = !DIFile(filename: "test.c", directory: "F:\\tmp", checksumkind: CSK_MD5, checksum: "5fbd15e58dd6931fc3081de308d52889")
!7 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !8)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{i32 2, !"CodeView", i32 1}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 2}
!12 = !{i32 1, !"min_enum_size", i32 4}
!13 = !{!"clang version 12.0.0 (https://github.com/llvm/llvm-project.git fc031d29bea856f2b91a250fd81c5f9fb79dbe07)"}
!14 = distinct !DISubprogram(name: "f", scope: !6, file: !6, line: 9, type: !15, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!15 = !DISubroutineType(types: !16)
!16 = !{null, !8}
!17 = !DILocalVariable(name: "p", arg: 1, scope: !14, file: !6, line: 9, type: !8)
!18 = !DILocation(line: 9, scope: !14)
!19 = !DILocation(line: 10, scope: !14)
!20 = !DILocation(line: 0, scope: !21)
!21 = distinct !DILexicalBlock(scope: !14, file: !6, line: 10)
!22 = !DILocalVariable(name: "a", scope: !23, file: !6, line: 11, type: !8)
!23 = distinct !DILexicalBlock(scope: !21, file: !6, line: 10)
!24 = !DILocation(line: 11, scope: !23)
!25 = !DILocalVariable(name: "b", scope: !23, file: !6, line: 12, type: !8)
!26 = !DILocation(line: 12, scope: !23)
!27 = !DILocation(line: 13, scope: !23)
!28 = !DILocation(line: 14, scope: !23)
!29 = !DILocalVariable(name: "c", scope: !30, file: !6, line: 15, type: !8)
!30 = distinct !DILexicalBlock(scope: !21, file: !6, line: 14)
!31 = !DILocation(line: 15, scope: !30)
!32 = !DILocation(line: 16, scope: !30)
!33 = !DILocation(line: 17, scope: !30)
!34 = !DILocation(line: 18, scope: !14)
!35 = distinct !DISubprogram(name: "inlineinc", scope: !6, file: !6, line: 4, type: !36, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !4)
!36 = !DISubroutineType(types: !37)
!37 = !{!8, !8}
!38 = !DILocalVariable(name: "a", arg: 1, scope: !35, file: !6, line: 4, type: !8)
!39 = !DILocation(line: 4, scope: !35)
!40 = !DILocalVariable(name: "b", scope: !35, file: !6, line: 5, type: !8)
!41 = !DILocation(line: 5, scope: !35)
!42 = !DILocation(line: 6, scope: !35)
!43 = !DILocation(line: 7, scope: !35)
