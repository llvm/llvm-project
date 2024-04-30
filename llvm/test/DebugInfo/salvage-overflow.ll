; RUN: opt %s -passes='sroa,early-cse' -S | FileCheck %s
; RUN: opt --try-experimental-debuginfo-iterators %s -passes='sroa,early-cse' -S | FileCheck %s
; CHECK: DIExpression(DW_OP_constu, 9223372036854775808, DW_OP_minus, DW_OP_stack_value)
; Created from the following C input (and then delta-reduced the IR):
;
; extern unsigned long long use(unsigned long long);
; void f(unsigned long long x) {
;   for (; x > 0; x --) {
;     unsigned long long y = x + 0x8000000000000000;
;     use(x);
;   }
; }

define void @f(i64 noundef %x) #0 !dbg !9 {
entry:
  %x.addr = alloca i64, align 8
  %y = alloca i64, align 8
  br label %for.cond
for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i64, ptr %x.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %y, metadata !15, metadata !DIExpression())
, !dbg !29
  %1 = load i64, ptr %x.addr, align 8
  %add = add i64 %1, -9223372036854775808
  store i64 %add, ptr %y, align 8
  br label %for.cond
}
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

!llvm.module.flags = !{!3,!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None, sysroot: "/")
!1 = !DIFile(filename: "t.c", directory: "/")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 7, !"frame-pointer", i32 2}
!9 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 2, type: !10, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13)
!10 = !DISubroutineType(types: !11)
!11 = !{null, !12}
!12 = !DIBasicType(name: "unsigned long long", size: 64, encoding: DW_ATE_unsigned)
!13 = !{}
!15 = !DILocalVariable(name: "y", scope: !16, file: !1, line: 4, type: !12)
!16 = distinct !DILexicalBlock(scope: !17, file: !1, line: 3, column: 23)
!17 = distinct !DILexicalBlock(scope: !18, file: !1, line: 3, column: 3)
!18 = distinct !DILexicalBlock(scope: !9, file: !1, line: 3, column: 3)
!29 = !DILocation(line: 4, column: 24, scope: !16)
