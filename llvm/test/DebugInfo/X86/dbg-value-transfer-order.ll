; RUN: llc < %s | FileCheck %s

; Reduced manually from C source:
; unsigned __attribute__((noinline, optnone)) useslot(unsigned v) { return v; }
; void __attribute__((noinline)) f(unsigned cell_offset, unsigned *page_start_) {
;   unsigned cell = page_start_[0];
;   while (cell) {
;     unsigned bit_offset = cell_offset ? __builtin_ctz(cell) : 32;
;     unsigned bit_mask = 1U << bit_offset;
;     unsigned slot = (cell_offset + bit_offset) << /*log2(sizeof(void*))*/ 3;
;     cell ^= bit_mask + page_start_[slot];
;   }
; }
; int main() {
;   static unsigned pages[32] = {1, 2, 3, 4, 5, 6, 7};
;   f(3, &pages[0]);
; }

; We had a bug where the DBG_VALUE instruction for bit_offset would be emitted
; at the end of the basic block, long after its actual program point. What's
; interesting in this example is that the !range metadata produces an AssertSext
; DAG node that gets replaced during ISel. This leads to an unordered SDDbgValue
; vector, which has to be sorted by IR order before it is processed in parallel
; with the Orders insertion point vector.

; CHECK-LABEL: f: # @f
; CHECK: .LBB0_2:                                # %while.body
; CHECK:         movl    $32, %ecx
; CHECK:         testl   {{.*}}
; CHECK:         jne     .LBB0_4
; CHECK: # %bb.3:                                 # %if.then
; CHECK:         callq   if_then
; CHECK:         movl    %eax, %ecx
; CHECK: .LBB0_4:                                # %if.end
;        Check that this DEBUG_VALUE comes before the left shift.
; CHECK:         #DEBUG_VALUE: bit_offset <- $ecx
; CHECK:         .cv_loc 0 1 8 28                # t.c:8:28
; CHECK:         movl    $1, %[[reg:[^ ]*]]
; CHECK:         shll    %cl, %[[reg]]

; ModuleID = 't.c'
source_filename = "t.c"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24215"

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare i32 @if_then()

; Function Attrs: noinline nounwind readonly uwtable
define void @f(i32 %cell_offset, i32* nocapture %page_start_) local_unnamed_addr #0 !dbg !31 {
entry:
  %0 = load i32, i32* %page_start_, align 4, !dbg !45
  %tobool14 = icmp eq i32 %0, 0, !dbg !47
  br i1 %tobool14, label %while.end, label %while.body, !dbg !47

while.body:                                       ; preds = %while.body.lr.ph, %while.body
  %cell.015 = phi i32 [ %0, %entry ], [ %xor, %if.end ]
  %tobool1 = icmp eq i32 %cell_offset, 0
  br i1 %tobool1, label %if.then, label %if.end

if.then:
  %1 = call i32 @if_then(), !dbg !48, !range !49
  br label %if.end

if.end:
  %cond = phi i32 [ %1, %if.then ], [ 32, %while.body ]
  tail call void @llvm.dbg.value(metadata i32 %cond, metadata !39, metadata !DIExpression()), !dbg !51
  %shl = shl i32 1, %cond, !dbg !52
  %add = add i32 %cond, %cell_offset, !dbg !54
  %shl2 = shl i32 %add, 3, !dbg !55
  %idxprom = zext i32 %shl2 to i64, !dbg !57
  %arrayidx3 = getelementptr inbounds i32, i32* %page_start_, i64 %idxprom, !dbg !57
  %2 = load i32, i32* %arrayidx3, align 4, !dbg !57
  %add4 = add i32 %2, %shl, !dbg !58
  %xor = xor i32 %add4, %cell.015, !dbg !59
  tail call void @llvm.dbg.value(metadata i32 %xor, metadata !38, metadata !DIExpression()), !dbg !46
  %tobool = icmp eq i32 %xor, 0, !dbg !47
  br i1 %tobool, label %while.end, label %while.body, !dbg !47, !llvm.loop !60

while.end:                                        ; preds = %while.body, %entry
  ret void, !dbg !62
}

; Function Attrs: nounwind readnone speculatable
declare i32 @llvm.cttz.i32(i32, i1) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind readonly uwtable }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!14, !15, !16, !17}
!llvm.ident = !{!18}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3)
!1 = !DIFile(filename: "t.c", directory: "C:\5Csrc\5Cllvm-project\5Cbuild", checksumkind: CSK_MD5, checksum: "f80d0003faf76554dfdec6c95da285cc")
!2 = !{}
!3 = !{!4}
!4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression())
!5 = distinct !DIGlobalVariable(name: "pages", scope: !6, file: !1, line: 18, type: !10, isLocal: true, isDefinition: true)
!6 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 17, type: !7, isLocal: false, isDefinition: true, scopeLine: 17, isOptimized: true, unit: !0, variables: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{!9}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, size: 1024, elements: !12)
!11 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!12 = !{!13}
!13 = !DISubrange(count: 32)
!14 = !{i32 2, !"CodeView", i32 1}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{i32 1, !"wchar_size", i32 2}
!17 = !{i32 7, !"PIC Level", i32 2}
!18 = !{!"clang version 6.0.0 "}
!31 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 4, type: !32, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !35)
!32 = !DISubroutineType(types: !33)
!33 = !{null, !11, !34}
!34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!35 = !{!36, !37, !38, !39, !41, !42}
!36 = !DILocalVariable(name: "page_start_", arg: 2, scope: !31, file: !1, line: 4, type: !34)
!37 = !DILocalVariable(name: "cell_offset", arg: 1, scope: !31, file: !1, line: 4, type: !11)
!38 = !DILocalVariable(name: "cell", scope: !31, file: !1, line: 5, type: !11)
!39 = !DILocalVariable(name: "bit_offset", scope: !40, file: !1, line: 7, type: !11)
!40 = distinct !DILexicalBlock(scope: !31, file: !1, line: 6, column: 16)
!41 = !DILocalVariable(name: "bit_mask", scope: !40, file: !1, line: 8, type: !11)
!42 = !DILocalVariable(name: "slot", scope: !40, file: !1, line: 9, type: !11)
!43 = !DILocation(line: 4, column: 66, scope: !31)
!44 = !DILocation(line: 4, column: 43, scope: !31)
!45 = !DILocation(line: 5, column: 19, scope: !31)
!46 = !DILocation(line: 5, column: 12, scope: !31)
!47 = !DILocation(line: 6, column: 3, scope: !31)
!48 = !DILocation(line: 7, column: 41, scope: !40)
!49 = !{i32 0, i32 33}
!50 = !DILocation(line: 7, column: 27, scope: !40)
!51 = !DILocation(line: 7, column: 14, scope: !40)
!52 = !DILocation(line: 8, column: 28, scope: !40)
!53 = !DILocation(line: 8, column: 14, scope: !40)
!54 = !DILocation(line: 9, column: 34, scope: !40)
!55 = !DILocation(line: 9, column: 48, scope: !40)
!56 = !DILocation(line: 9, column: 14, scope: !40)
!57 = !DILocation(line: 13, column: 24, scope: !40)
!58 = !DILocation(line: 13, column: 22, scope: !40)
!59 = !DILocation(line: 13, column: 10, scope: !40)
!60 = distinct !{!60, !47, !61}
!61 = !DILocation(line: 14, column: 3, scope: !31)
!62 = !DILocation(line: 15, column: 1, scope: !31)
