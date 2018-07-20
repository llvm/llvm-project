; RUN: llc < %s | FileCheck %s

; Make sure we insert DW_OP_deref when spilling indirect DBG_VALUE instructions.
; In this example, 'nt' is passed by address because it is not trivially
; copyable. When we spill the physical argument register at the barrier, we need
; to insert DW_OP_deref.

; #define FORCE_SPILL() \
;   __asm volatile("" : : : \
;                    "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "r8", \
;                    "r9", "r10", "r11", "r12", "r13", "r14", "r15")
; struct NonTrivial {
;   NonTrivial();
;   ~NonTrivial();
;   int i;
; };
; int foo(NonTrivial nt) {
;   FORCE_SPILL();
;   return nt.i;
; }

; CHECK-LABEL: _Z3foo10NonTrivial:
; CHECK: #DEBUG_VALUE: foo:nt <- [$rdi+0]
; CHECK: movq    %rdi, -8(%rsp)          # 8-byte Spill
; CHECK: #DEBUG_VALUE: foo:nt <- [DW_OP_constu 8, DW_OP_minus, DW_OP_deref] [$rsp+0]
; CHECK: #APP
; CHECK: #NO_APP
; CHECK: movq    -8(%rsp), %rax          # 8-byte Reload
; CHECK: movl    (%rax), %eax

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64--linux"

%struct.NonTrivial = type { i32 }

; Function Attrs: nounwind uwtable
define i32 @_Z3foo10NonTrivial(%struct.NonTrivial* nocapture readonly %nt) local_unnamed_addr #0 !dbg !7 {
entry:
  tail call void @llvm.dbg.declare(metadata %struct.NonTrivial* %nt, metadata !20, metadata !DIExpression()), !dbg !21
  tail call void asm sideeffect "", "~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{dirflag},~{fpsr},~{flags}"() #2, !dbg !22, !srcloc !23
  %i = getelementptr inbounds %struct.NonTrivial, %struct.NonTrivial* %nt, i64 0, i32 0, !dbg !24
  %0 = load i32, i32* %i, align 4, !dbg !24, !tbaa !25
  ret i32 %0, !dbg !30
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { nounwind }

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
!7 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foo10NonTrivial", scope: !1, file: !1, line: 10, type: !8, isLocal: false, isDefinition: true, scopeLine: 10, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !19)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !11}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "NonTrivial", file: !1, line: 5, size: 32, elements: !12, identifier: "_ZTS10NonTrivial")
!12 = !{!13, !14, !18}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !11, file: !1, line: 8, baseType: !10, size: 32)
!14 = !DISubprogram(name: "NonTrivial", scope: !11, file: !1, line: 6, type: !15, isLocal: false, isDefinition: false, scopeLine: 6, flags: DIFlagPrototyped, isOptimized: true)
!15 = !DISubroutineType(types: !16)
!16 = !{null, !17}
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!18 = !DISubprogram(name: "~NonTrivial", scope: !11, file: !1, line: 7, type: !15, isLocal: false, isDefinition: false, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: true)
!19 = !{!20}
!20 = !DILocalVariable(name: "nt", arg: 1, scope: !7, file: !1, line: 10, type: !11)
!21 = !DILocation(line: 10, column: 20, scope: !7)
!22 = !DILocation(line: 11, column: 3, scope: !7)
!23 = !{i32 -2147471481}
!24 = !DILocation(line: 12, column: 13, scope: !7)
!25 = !{!26, !27, i64 0}
!26 = !{!"_ZTS10NonTrivial", !27, i64 0}
!27 = !{!"int", !28, i64 0}
!28 = !{!"omnipotent char", !29, i64 0}
!29 = !{!"Simple C++ TBAA"}
!30 = !DILocation(line: 12, column: 3, scope: !7)
