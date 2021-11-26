; RUN: llc < %s -experimental-debug-variable-locations=false | FileCheck -check-prefixes=CHECK,OPT %s
; RUN: llc -O0 < %s -experimental-debug-variable-locations=false | FileCheck -check-prefixes=CHECK,OPTNONE %s
; RUN: llc < %s -experimental-debug-variable-locations=true | FileCheck -check-prefixes=CHECK,OPT %s
; RUN: llc -O0 < %s -experimental-debug-variable-locations=true | FileCheck -check-prefixes=CHECK,OPTNONE %s

; Make sure we insert DW_OP_deref when spilling indirect DBG_VALUE instructions.

; C++ source:
; #define FORCE_SPILL() \
;   __asm volatile("" : : : \
;                    "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "r8", \
;                    "r9", "r10", "r11", "r12", "r13", "r14", "r15")
; struct string {
;   string();
;   string(int i);
;   ~string();
;   int i = 0;
; };
; string get_string() {
;   string result = 3;
;   FORCE_SPILL();
;   return result;
; }

; CHECK-LABEL: _Z10get_stringv:

; OPT: #DEBUG_VALUE: get_string:result <- [$rdi+0]
; OPT: movq   %rdi, [[OFFS:[0-9]+]](%rsp)          # 8-byte Spill
; OPT: #DEBUG_VALUE: get_string:result <- [DW_OP_plus_uconst [[OFFS]], DW_OP_deref] [$rsp+0]
; OPT: callq  _ZN6stringC1Ei

; OPTNONE: #DEBUG_VALUE: get_string:result <- [DW_OP_deref] [$rsp+0]
; OPTNONE: movq   %rdi, %rax
; OPTNONE: movq   %rax, [[OFFS:[0-9]+]](%rsp)          # 8-byte Spill
; OPTNONE: #DEBUG_VALUE: get_string:result <- [$rdi+0]
; OPTNONE: callq  _ZN6stringC1Ei

; CHECK: #APP
; CHECK: #NO_APP

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64--linux"

%struct.string = type { i32 }

; Function Attrs: uwtable
define void @_Z10get_stringv(%struct.string* noalias sret(%struct.string) %agg.result) #0 !dbg !7 {
entry:
  %nrvo = alloca i1, align 1
  store i1 false, i1* %nrvo, align 1, !dbg !24
  call void @llvm.dbg.declare(metadata %struct.string* %agg.result, metadata !23, metadata !DIExpression()), !dbg !25
  call void @_ZN6stringC1Ei(%struct.string* %agg.result, i32 3), !dbg !26
  call void asm sideeffect "", "~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{dirflag},~{fpsr},~{flags}"() #3, !dbg !27, !srcloc !28
  store i1 true, i1* %nrvo, align 1, !dbg !29
  %nrvo.val = load i1, i1* %nrvo, align 1, !dbg !30
  br i1 %nrvo.val, label %nrvo.skipdtor, label %nrvo.unused, !dbg !30

nrvo.unused:                                      ; preds = %entry
  call void @_ZN6stringD1Ev(%struct.string* %agg.result), !dbg !30
  br label %nrvo.skipdtor, !dbg !30

nrvo.skipdtor:                                    ; preds = %nrvo.unused, %entry
  ret void, !dbg !30
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @_ZN6stringC1Ei(%struct.string*, i32) unnamed_addr

declare void @_ZN6stringD1Ev(%struct.string*) unnamed_addr

attributes #0 = { uwtable }
attributes #1 = { nounwind readnone speculatable }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 6.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.cpp", directory: "C:\5Csrc\5Cllvm-project\5Cbuild")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 6.0.0 "}
!7 = distinct !DISubprogram(name: "get_string", linkageName: "_Z10get_stringv", scope: !1, file: !1, line: 13, type: !8, isLocal: false, isDefinition: true, scopeLine: 13, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !22)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "string", file: !1, line: 7, size: 32, elements: !11, identifier: "_ZTS6string")
!11 = !{!12, !14, !18, !21}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !10, file: !1, line: 11, baseType: !13, size: 32)
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DISubprogram(name: "string", scope: !10, file: !1, line: 8, type: !15, isLocal: false, isDefinition: false, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: true)
!15 = !DISubroutineType(types: !16)
!16 = !{null, !17}
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!18 = !DISubprogram(name: "string", scope: !10, file: !1, line: 9, type: !19, isLocal: false, isDefinition: false, scopeLine: 9, flags: DIFlagPrototyped, isOptimized: true)
!19 = !DISubroutineType(types: !20)
!20 = !{null, !17, !13}
!21 = !DISubprogram(name: "~string", scope: !10, file: !1, line: 10, type: !15, isLocal: false, isDefinition: false, scopeLine: 10, flags: DIFlagPrototyped, isOptimized: true)
!22 = !{!23}
!23 = !DILocalVariable(name: "result", scope: !7, file: !1, line: 14, type: !10)
!24 = !DILocation(line: 14, column: 3, scope: !7)
!25 = !DILocation(line: 14, column: 10, scope: !7)
!26 = !DILocation(line: 14, column: 19, scope: !7)
!27 = !DILocation(line: 15, column: 3, scope: !7)
!28 = !{i32 -2147471175}
!29 = !DILocation(line: 16, column: 3, scope: !7)
!30 = !DILocation(line: 17, column: 1, scope: !7)
