; RUN: llc < %s -filetype=obj | llvm-readobj --codeview - | FileCheck %s --check-prefix=OBJ

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

; Function Attrs: noinline nounwind optnone
define dso_local arm_aapcs_vfpcc void @f(i32 %p) !dbg !14 {
entry:
  %p.addr = alloca i32, align 4
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %c = alloca i32, align 4
  store i32 %p, ptr %p.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %p.addr, metadata !17, metadata !DIExpression()), !dbg !18
  %0 = load i32, ptr %p.addr, align 4, !dbg !19
  %tobool = icmp ne i32 %0, 0, !dbg !19
  br i1 %tobool, label %if.then, label %if.else, !dbg !19

if.then:                                          ; preds = %entry
  call void @llvm.dbg.declare(metadata ptr %a, metadata !20, metadata !DIExpression()), !dbg !23
  %call = call arm_aapcs_vfpcc i32 @getint(), !dbg !23
  store i32 %call, ptr %a, align 4, !dbg !23
  call void @llvm.dbg.declare(metadata ptr %b, metadata !24, metadata !DIExpression()), !dbg !25
  %1 = load i32, ptr %a, align 4, !dbg !25
  %call1 = call arm_aapcs_vfpcc i32 @inlineinc(i32 %1), !dbg !25
  store i32 %call1, ptr %b, align 4, !dbg !25
  %2 = load i32, ptr %b, align 4, !dbg !26
  call arm_aapcs_vfpcc void @putint(i32 %2), !dbg !26
  br label %if.end, !dbg !27

if.else:                                          ; preds = %entry
  call void @llvm.dbg.declare(metadata ptr %c, metadata !28, metadata !DIExpression()), !dbg !30
  %call2 = call arm_aapcs_vfpcc i32 @getint(), !dbg !30
  store i32 %call2, ptr %c, align 4, !dbg !30
  %3 = load i32, ptr %c, align 4, !dbg !31
  call arm_aapcs_vfpcc void @putint(i32 %3), !dbg !31
  br label %if.end, !dbg !32

if.end:                                           ; preds = %if.else, %if.then
  ret void, !dbg !33
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata)

declare dso_local arm_aapcs_vfpcc i32 @getint()

; Function Attrs: noinline nounwind optnone
define internal arm_aapcs_vfpcc i32 @inlineinc(i32 %a) !dbg !34 {
entry:
  %a.addr = alloca i32, align 4
  %b = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %a.addr, metadata !37, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.declare(metadata ptr %b, metadata !39, metadata !DIExpression()), !dbg !40
  %0 = load i32, ptr %a.addr, align 4, !dbg !40
  %add = add nsw i32 %0, 1, !dbg !40
  store i32 %add, ptr %b, align 4, !dbg !40
  %1 = load volatile i32, ptr @x, align 4, !dbg !41
  %inc = add nsw i32 %1, 1, !dbg !41
  store volatile i32 %inc, ptr @x, align 4, !dbg !41
  %2 = load i32, ptr %b, align 4, !dbg !42
  ret i32 %2, !dbg !42
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
!20 = !DILocalVariable(name: "a", scope: !21, file: !6, line: 11, type: !8)
!21 = distinct !DILexicalBlock(scope: !22, file: !6, line: 10)
!22 = distinct !DILexicalBlock(scope: !14, file: !6, line: 10)
!23 = !DILocation(line: 11, scope: !21)
!24 = !DILocalVariable(name: "b", scope: !21, file: !6, line: 12, type: !8)
!25 = !DILocation(line: 12, scope: !21)
!26 = !DILocation(line: 13, scope: !21)
!27 = !DILocation(line: 14, scope: !21)
!28 = !DILocalVariable(name: "c", scope: !29, file: !6, line: 15, type: !8)
!29 = distinct !DILexicalBlock(scope: !22, file: !6, line: 14)
!30 = !DILocation(line: 15, scope: !29)
!31 = !DILocation(line: 16, scope: !29)
!32 = !DILocation(line: 17, scope: !29)
!33 = !DILocation(line: 18, scope: !14)
!34 = distinct !DISubprogram(name: "inlineinc", scope: !6, file: !6, line: 4, type: !35, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !4)
!35 = !DISubroutineType(types: !36)
!36 = !{!8, !8}
!37 = !DILocalVariable(name: "a", arg: 1, scope: !34, file: !6, line: 4, type: !8)
!38 = !DILocation(line: 4, scope: !34)
!39 = !DILocalVariable(name: "b", scope: !34, file: !6, line: 5, type: !8)
!40 = !DILocation(line: 5, scope: !34)
!41 = !DILocation(line: 6, scope: !34)
!42 = !DILocation(line: 7, scope: !34)
