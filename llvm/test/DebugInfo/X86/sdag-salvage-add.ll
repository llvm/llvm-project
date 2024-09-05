; RUN: llc -mtriple=x86_64-unknown-unknown -stop-before livedebugvalues %s -o -\
; RUN:     -experimental-debug-variable-locations=false \
; RUN:   | FileCheck %s --check-prefixes=CHECK,DBGVALUE
; RUN: llc -mtriple=x86_64-unknown-unknown -stop-before livedebugvalues %s -o -\
; RUN:     -experimental-debug-variable-locations=true \
; RUN:   | FileCheck %s --check-prefixes=CHECK,INSTRREF
;
; Generated at -O1 from:
; typedef struct {
;   unsigned long long c;
; } S1;
; struct S3 {
;   unsigned long long packed;
; };
; struct S6 {
;   struct S0 *b;
; };
; void f(struct S3 *a3)
; {
;   struct S4 *s4 = (struct S4 *)(a3->packed + 0x1000UL);
;   struct S6 *myVar = (struct S6 *)s4;
;   struct S0 *b = myVar->b;
;   use(b);
; }
;
; The debug info is attached to the ADD 4096 operation, which doesn't survive
; instruction selection as it is folded into the load. As a result, we should
; refer to s4 and myVar with complex expressions.
;
; NB: instruction referencing and DBG_VALUE modes produce debug insts in a
; different order.
;
; CHECK:         ![[S4:.*]] = !DILocalVariable(name: "s4",
; CHECK:         ![[MYVAR:.*]] = !DILocalVariable(name: "myVar",
; CHECK:         $rax = MOV64rm
; INSTRREF-SAME: debug-instr-number 2,
; INSTRREF-NEXT: DBG_INSTR_REF ![[S4]],
; DBGVALUE-NEXT: DBG_VALUE $rax, $noreg, ![[MYVAR]],
; DBGVALUE-SAME:       !DIExpression(DW_OP_plus_uconst, 4096, DW_OP_stack_value)
; INSTRREF-SAME:       !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_plus_uconst, 4096, DW_OP_stack_value)
; INSTRREF-SAME: dbg-instr-ref(2, 0)

; INSTRREF:      DBG_INSTR_REF ![[MYVAR]],
; DBGVALUE:      DBG_VALUE $rax, $noreg, ![[S4]],
; DBGVALUE-SAME:           !DIExpression(DW_OP_plus_uconst, 4096, DW_OP_stack_value)
; INSTRREF-SAME:           !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_plus_uconst, 4096, DW_OP_stack_value)
; INSTRREF-SAME: dbg-instr-ref(2, 0)
; CHECK-NEXT: $rdi = MOV64rm killed renamable $rax, 1, $noreg, 4096, $noreg,

source_filename = "test.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.13.0"

%struct.S3 = type { i64 }
%struct.S4 = type opaque
%struct.S0 = type opaque

; Function Attrs: noinline nounwind ssp uwtable
define void @f(ptr nocapture readonly %a3) local_unnamed_addr #0 !dbg !6 {
entry:
  tail call void @llvm.dbg.value(metadata ptr %a3, metadata !15, metadata !DIExpression()), !dbg !30
  %0 = load i64, ptr %a3, align 8, !dbg !31
  %add = add i64 %0, 4096, !dbg !37
  %1 = inttoptr i64 %add to ptr, !dbg !38
  tail call void @llvm.dbg.value(metadata ptr %1, metadata !16, metadata !DIExpression()), !dbg !39
  tail call void @llvm.dbg.value(metadata ptr %1, metadata !17, metadata !DIExpression()), !dbg !40
  %2 = load ptr, ptr %1, align 8, !dbg !41
  tail call void @llvm.dbg.value(metadata ptr %2, metadata !24, metadata !DIExpression()), !dbg !45
  %call = tail call i32 (ptr, ...) @use(ptr %2) #3, !dbg !46
  ret void, !dbg !47
}

declare i32 @use(...) local_unnamed_addr

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { noinline nounwind ssp uwtable }
attributes #2 = { nounwind readnone speculatable }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!25, !26, !27, !28}
!llvm.ident = !{!29}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 (trunk 316467) (llvm/trunk 316466)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{}
!3 = !{!4, !18}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!5 = !DICompositeType(tag: DW_TAG_structure_type, name: "S4", scope: !6, file: !1, line: 20, flags: DIFlagFwdDecl)
!6 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 18, type: !7, isLocal: false, isDefinition: true, scopeLine: 19, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !14)
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9}
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S3", file: !1, line: 5, size: 64, elements: !11)
!11 = !{!12}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "packed", scope: !10, file: !1, line: 6, baseType: !13, size: 64)
!13 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!14 = !{!15, !16, !17, !24}
!15 = !DILocalVariable(name: "a3", arg: 1, scope: !6, file: !1, line: 18, type: !9)
!16 = !DILocalVariable(name: "s4", scope: !6, file: !1, line: 20, type: !4)
!17 = !DILocalVariable(name: "myVar", scope: !6, file: !1, line: 21, type: !18)
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !19, size: 64)
!19 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S6", file: !1, line: 8, size: 64, elements: !20)
!20 = !{!21}
!21 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !19, file: !1, line: 9, baseType: !22, size: 64)
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !23, size: 64)
!23 = !DICompositeType(tag: DW_TAG_structure_type, name: "S0", file: !1, line: 4, flags: DIFlagFwdDecl)
!24 = !DILocalVariable(name: "b", scope: !6, file: !1, line: 22, type: !22)
!25 = !{i32 2, !"Dwarf Version", i32 4}
!26 = !{i32 2, !"Debug Info Version", i32 3}
!27 = !{i32 1, !"wchar_size", i32 4}
!28 = !{i32 7, !"PIC Level", i32 2}
!29 = !{!"clang version 6.0.0 (trunk 316467) (llvm/trunk 316466)"}
!30 = !DILocation(line: 18, column: 14, scope: !6)
!31 = !DILocation(line: 20, column: 37, scope: !6)
!37 = !DILocation(line: 20, column: 44, scope: !6)
!38 = !DILocation(line: 20, column: 19, scope: !6)
!39 = !DILocation(line: 20, column: 14, scope: !6)
!40 = !DILocation(line: 21, column: 14, scope: !6)
!41 = !DILocation(line: 22, column: 25, scope: !6)
!45 = !DILocation(line: 22, column: 14, scope: !6)
!46 = !DILocation(line: 23, column: 3, scope: !6)
!47 = !DILocation(line: 24, column: 1, scope: !6)
