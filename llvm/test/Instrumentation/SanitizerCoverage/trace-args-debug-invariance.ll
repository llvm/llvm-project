; Debug information may enrich how an observation is interpreted, but must never
; decide what is observed: instrumenting this module and then stripping its debug
; information gives exactly the module that stripping first and instrumenting
; then gives.
;
; The input carries everything that could tempt the pass: source-level
; parameter records that disagree with the IR argument list (an sret pointer, a
; struct coerced into one register and split into two fragments, a first-class
; aggregate one of whose fragments was optimized out), a struct pointer with a
; known field layout, a #dbg_declare of a byval argument, an unoptimized
; function that stores its arguments to declared allocas, and an inlined callee
; with parameters of its own.
;
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=3 -sanitizer-coverage-trace-pc-guard -sanitizer-coverage-trace-args -sanitizer-coverage-trace-ret \
; RUN:   | opt -strip-debug -S -o %t.instrumented-then-stripped.ll
; RUN: opt < %s -strip-debug \
; RUN:   | opt -passes='module(sancov-module)' -sanitizer-coverage-level=3 -sanitizer-coverage-trace-pc-guard -sanitizer-coverage-trace-args -sanitizer-coverage-trace-ret -S -o %t.stripped-then-instrumented.ll
; RUN: diff %t.instrumented-then-stripped.ll %t.stripped-then-instrumented.ll

; Debug information still gives the calls a location: the function's own
; opening line, even where the entry block starts with an inlined callee.
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=1 -sanitizer-coverage-trace-args -S | FileCheck %s
; CHECK-LABEL: define i32 @caller(
; CHECK: call void @__sanitizer_cov_trace_args(i64 8, i64 %n, ptr null) #{{[0-9]+}}, !dbg ![[ENTRY:[0-9]+]]
; CHECK-NOT: call void @__sanitizer_cov_trace_args
; CHECK: ret i32
; CHECK-DAG: ![[ENTRY]] = !DILocation(line: 10, scope: ![[CALLER:[0-9]+]])
; CHECK-DAG: ![[CALLER]] = distinct !DISubprogram(name: "caller"

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S = type { i64, i64, i64 }

; struct S make(struct pair p, struct S *q) with struct pair { int a, b; }
define void @make(ptr sret(%struct.S) %r, i64 noundef %p, ptr noundef %q) !dbg !10 {
entry:
  %a = trunc i64 %p to i32
  %hi = lshr i64 %p, 32
  %b = trunc i64 %hi to i32
    #dbg_value(i32 %a, !13, !DIExpression(DW_OP_LLVM_fragment, 0, 32), !15)
    #dbg_value(i32 %b, !13, !DIExpression(DW_OP_LLVM_fragment, 32, 32), !15)
    #dbg_value(ptr %q, !14, !DIExpression(), !15)
  store i64 %p, ptr %r, !dbg !15
  ret void, !dbg !15
}

; long use(struct S s), passed in memory and described by a #dbg_declare.
define i64 @use(ptr byval(%struct.S) %s) !dbg !20 {
entry:
    #dbg_declare(ptr %s, !21, !DIExpression(), !22)
  %v = load i64, ptr %s, !dbg !22
  ret i64 %v, !dbg !22
}

; long caller(long n) { return leaf(n * 2); } with leaf(long m) inlined.
define i32 @caller(i64 noundef %n) !dbg !30 {
entry:
  %m = shl i64 %n, 1, !dbg !34
    #dbg_value(i64 %n, !31, !DIExpression(), !33)
    #dbg_value(i64 %m, !35, !DIExpression(), !34)
  %t = trunc i64 %m to i32, !dbg !34
  ret i32 %t, !dbg !33
}

; long split(struct mix m, int z), with m passed as [2 x i64] (AArch64).
define i64 @split([2 x i64] noundef %m, i32 noundef %z) !dbg !50 {
entry:
  %l = extractvalue [2 x i64] %m, 1
    #dbg_value(i64 poison, !51, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !53)
    #dbg_value(i64 %l, !51, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !53)
    #dbg_value(i32 %z, !52, !DIExpression(), !53)
  %zl = sext i32 %z to i64
  %r = add i64 %l, %zl
  ret i64 %r, !dbg !53
}

; The same at -O0, where the arguments are stored to declared allocas.
define i64 @split_o0([2 x i64] noundef %m, i32 noundef %z) !dbg !60 {
entry:
  %m.addr = alloca [2 x i64], align 8
  %z.addr = alloca i32, align 4
  store [2 x i64] %m, ptr %m.addr, align 8
    #dbg_declare(ptr %m.addr, !61, !DIExpression(), !63)
  store i32 %z, ptr %z.addr, align 4
    #dbg_declare(ptr %z.addr, !62, !DIExpression(), !63)
  %p = getelementptr inbounds i8, ptr %m.addr, i64 8
  %l = load i64, ptr %p, align 8, !dbg !63
  ret i64 %l, !dbg !63
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, isOptimized: true, emissionKind: FullDebug)
!1 = !DIFile(filename: "invariance.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!6 = !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !1, size: 192, elements: !7)
!7 = !{!40, !41, !42}
!8 = !DICompositeType(tag: DW_TAG_structure_type, name: "pair", file: !1, size: 64, elements: !{})
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64)
!10 = distinct !DISubprogram(name: "make", scope: !1, file: !1, line: 1, type: !11, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !DISubroutineType(types: !12)
!12 = !{!6, !8, !9}
!13 = !DILocalVariable(name: "p", arg: 1, scope: !10, file: !1, line: 1, type: !8)
!14 = !DILocalVariable(name: "q", arg: 2, scope: !10, file: !1, line: 1, type: !9)
!15 = !DILocation(line: 1, column: 1, scope: !10)
!20 = distinct !DISubprogram(name: "use", scope: !1, file: !1, line: 5, type: !23, scopeLine: 5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!21 = !DILocalVariable(name: "s", arg: 1, scope: !20, file: !1, line: 5, type: !6)
!22 = !DILocation(line: 5, column: 1, scope: !20)
!23 = !DISubroutineType(types: !24)
!24 = !{!5, !6}
!30 = distinct !DISubprogram(name: "caller", scope: !1, file: !1, line: 10, type: !36, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!31 = !DILocalVariable(name: "n", arg: 1, scope: !30, file: !1, line: 10, type: !5)
!32 = distinct !DISubprogram(name: "leaf", scope: !1, file: !1, line: 8, type: !36, scopeLine: 8, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!33 = !DILocation(line: 10, column: 1, scope: !30)
!34 = !DILocation(line: 8, column: 1, scope: !32, inlinedAt: !33)
!35 = !DILocalVariable(name: "m", arg: 1, scope: !32, file: !1, line: 8, type: !5)
!36 = !DISubroutineType(types: !37)
!37 = !{!5, !5}
!40 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !6, file: !1, baseType: !5, size: 64)
!41 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !6, file: !1, baseType: !5, size: 64, offset: 64)
!42 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !6, file: !1, baseType: !5, size: 64, offset: 128)
!43 = !DICompositeType(tag: DW_TAG_structure_type, name: "mix", file: !1, size: 128, elements: !{})
!44 = !DISubroutineType(types: !45)
!45 = !{!5, !43, !4}
!50 = distinct !DISubprogram(name: "split", scope: !1, file: !1, line: 20, type: !44, scopeLine: 20, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!51 = !DILocalVariable(name: "m", arg: 1, scope: !50, file: !1, line: 20, type: !43)
!52 = !DILocalVariable(name: "z", arg: 2, scope: !50, file: !1, line: 20, type: !4)
!53 = !DILocation(line: 20, column: 1, scope: !50)
!60 = distinct !DISubprogram(name: "split_o0", scope: !1, file: !1, line: 30, type: !44, scopeLine: 30, spFlags: DISPFlagDefinition, unit: !0)
!61 = !DILocalVariable(name: "m", arg: 1, scope: !60, file: !1, line: 30, type: !43)
!62 = !DILocalVariable(name: "z", arg: 2, scope: !60, file: !1, line: 30, type: !4)
!63 = !DILocation(line: 30, column: 1, scope: !60)
