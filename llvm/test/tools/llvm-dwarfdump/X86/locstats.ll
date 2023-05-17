; RUN: llc -debug-entry-values %s -o - -filetype=obj \
; RUN:   | llvm-dwarfdump -statistics - | FileCheck %s

; CHECK:      "sum_all_variables(#bytes in parent scope covered by DW_OP_entry_value)": 5,
; CHECK-NEXT: "sum_all_params(#bytes in parent scope)": 20,
; CHECK-NEXT: "sum_all_params(#bytes in parent scope covered by DW_AT_location)": 20,
; CHECK-NEXT: "sum_all_params(#bytes in parent scope covered by DW_OP_entry_value)": 5,
; CHECK-NEXT: "sum_all_local_vars(#bytes in parent scope)": 90,
; CHECK-NEXT: "sum_all_local_vars(#bytes in parent scope covered by DW_AT_location)": 60,
; CHECK-NEXT: "sum_all_local_vars(#bytes in parent scope covered by DW_OP_entry_value)": 0,
; CHECK-NEXT: "#bytes within functions": 45,
; CHECK-NEXT: "#bytes within inlined functions": 0,
; CHECK-NEXT: "#params": 2,
; CHECK-NEXT: "#params with source location": 2,
; CHECK-NEXT: "#params with type": 2,
; CHECK-NEXT: "#params with binary location": 2,
; CHECK:      "#variables processed by location statistics": 6,
; CHECK-NEXT: "#variables with 0% of parent scope covered by DW_AT_location": 1,
; CHECK-NEXT: "#variables with (0%,10%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#variables with [10%,20%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#variables with [20%,30%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#variables with [30%,40%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#variables with [40%,50%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#variables with [50%,60%) of parent scope covered by DW_AT_location": 1,
; CHECK-NEXT: "#variables with [60%,70%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#variables with [70%,80%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#variables with [80%,90%) of parent scope covered by DW_AT_location": 1,
; CHECK-NEXT: "#variables with [90%,100%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#variables with 100% of parent scope covered by DW_AT_location": 3,
; CHECK-NEXT: "#variables - entry values with 0% of parent scope covered by DW_AT_location": 1,
; CHECK-NEXT: "#variables - entry values with (0%,10%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#variables - entry values with [10%,20%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#variables - entry values with [20%,30%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#variables - entry values with [30%,40%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#variables - entry values with [40%,50%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#variables - entry values with [50%,60%) of parent scope covered by DW_AT_location": 2,
; CHECK-NEXT: "#variables - entry values with [60%,70%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#variables - entry values with [70%,80%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#variables - entry values with [80%,90%) of parent scope covered by DW_AT_location": 1,
; CHECK-NEXT: "#variables - entry values with [90%,100%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#variables - entry values with 100% of parent scope covered by DW_AT_location": 2,
; CHECK-NEXT: "#params processed by location statistics": 2,
; CHECK-NEXT: "#params with 0% of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#params with (0%,10%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#params with [10%,20%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#params with [20%,30%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#params with [30%,40%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#params with [40%,50%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#params with [50%,60%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#params with [60%,70%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#params with [70%,80%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#params with [80%,90%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#params with [90%,100%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#params with 100% of parent scope covered by DW_AT_location": 2,
; CHECK-NEXT: "#params - entry values with 0% of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#params - entry values with (0%,10%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#params - entry values with [10%,20%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#params - entry values with [20%,30%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#params - entry values with [30%,40%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#params - entry values with [40%,50%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#params - entry values with [50%,60%) of parent scope covered by DW_AT_location": 1,
; CHECK-NEXT: "#params - entry values with [60%,70%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#params - entry values with [70%,80%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#params - entry values with [80%,90%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#params - entry values with [90%,100%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#params - entry values with 100% of parent scope covered by DW_AT_location": 1,
; CHECK-NEXT: "#local vars processed by location statistics": 4,
; CHECK-NEXT: "#local vars with 0% of parent scope covered by DW_AT_location": 1,
; CHECK-NEXT: "#local vars with (0%,10%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#local vars with [10%,20%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#local vars with [20%,30%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#local vars with [30%,40%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#local vars with [40%,50%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#local vars with [50%,60%) of parent scope covered by DW_AT_location": 1,
; CHECK-NEXT: "#local vars with [60%,70%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#local vars with [70%,80%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#local vars with [80%,90%) of parent scope covered by DW_AT_location": 1,
; CHECK-NEXT: "#local vars with [90%,100%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#local vars with 100% of parent scope covered by DW_AT_location": 1,
; CHECK-NEXT: "#local vars - entry values with 0% of parent scope covered by DW_AT_location": 1,
; CHECK-NEXT: "#local vars - entry values with (0%,10%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#local vars - entry values with [10%,20%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#local vars - entry values with [20%,30%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#local vars - entry values with [30%,40%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#local vars - entry values with [40%,50%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#local vars - entry values with [50%,60%) of parent scope covered by DW_AT_location": 1,
; CHECK-NEXT: "#local vars - entry values with [60%,70%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#local vars - entry values with [70%,80%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#local vars - entry values with [80%,90%) of parent scope covered by DW_AT_location": 1,
; CHECK-NEXT: "#local vars - entry values with [90%,100%) of parent scope covered by DW_AT_location": 0,
; CHECK-NEXT: "#local vars - entry values with 100% of parent scope covered by DW_AT_location": 1

; The source code of the test case:
; extern void fn3(int *);
; extern void fn2 (int);
; __attribute__((noinline))
; void
; fn1 (int x, int y)
; {
;   int u = x + y;
;   if (x > 1)
;     u += 1;
;   else
;     u += 2;
;   if (y > 4)
;     u += x;
;   int a = 7;
;   fn2 (a);
;   u --;
; }

; __attribute__((noinline))
; int f()
; {
;   int l, k;
;   fn3(&l);
;   fn3(&k);
;   fn1 (l, k);
;   return 0;
; }

; ModuleID = 'test.c'
source_filename = "test.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local void @fn1(i32 %x, i32 %y) local_unnamed_addr !dbg !16 {
entry:
  call void @llvm.dbg.value(metadata i32 %x, metadata !20, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 %y, metadata !21, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 undef, metadata !22, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 undef, metadata !22, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 7, metadata !23, metadata !DIExpression()), !dbg !24
  tail call void @fn2(i32 7), !dbg !25
  call void @llvm.dbg.value(metadata i32 undef, metadata !22, metadata !DIExpression(DW_OP_constu, 1, DW_OP_minus, DW_OP_stack_value)), !dbg !24
  ret void, !dbg !26
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)

declare !dbg !4 dso_local void @fn2(i32) local_unnamed_addr

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @f() local_unnamed_addr !dbg !27 {
entry:
  %l = alloca i32, align 4
  %k = alloca i32, align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %l), !dbg !33
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %k), !dbg !33
  call void @llvm.dbg.value(metadata ptr %l, metadata !31, metadata !DIExpression(DW_OP_deref)), !dbg !34
  call void @fn3(ptr nonnull %l), !dbg !35
  call void @llvm.dbg.value(metadata ptr %k, metadata !32, metadata !DIExpression(DW_OP_deref)), !dbg !34
  call void @fn3(ptr nonnull %k), !dbg !36
  %0 = load i32, ptr %l, align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %0, metadata !31, metadata !DIExpression()), !dbg !34
  %1 = load i32, ptr %k, align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %1, metadata !32, metadata !DIExpression()), !dbg !34
  call void @fn1(i32 %0, i32 %1), !dbg !37
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %k), !dbg !37
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %l), !dbg !37
  ret i32 0, !dbg !37
}

declare !dbg !8 dso_local void @fn3(ptr) local_unnamed_addr

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13, !14}
!llvm.ident = !{!15}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{}
!3 = !{!4, !8}
!4 = !DISubprogram(name: "fn2", scope: !1, file: !1, line: 2, type: !5, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DISubprogram(name: "fn3", scope: !1, file: !1, line: 1, type: !9, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!12 = !{i32 2, !"Dwarf Version", i32 4}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{!"clang version 10.0.0"}
!16 = distinct !DISubprogram(name: "fn1", scope: !1, file: !1, line: 6, type: !17, scopeLine: 7, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !19)
!17 = !DISubroutineType(types: !18)
!18 = !{null, !7, !7}
!19 = !{!20, !21, !22, !23}
!20 = !DILocalVariable(name: "x", arg: 1, scope: !16, file: !1, line: 6, type: !7)
!21 = !DILocalVariable(name: "y", arg: 2, scope: !16, file: !1, line: 6, type: !7)
!22 = !DILocalVariable(name: "u", scope: !16, file: !1, line: 8, type: !7)
!23 = !DILocalVariable(name: "a", scope: !16, file: !1, line: 18, type: !7)
!24 = !DILocation(line: 0, scope: !16)
!25 = !DILocation(line: 20, column: 3, scope: !16)
!26 = !DILocation(line: 22, column: 1, scope: !16)
!27 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 25, type: !28, scopeLine: 26, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !30)
!28 = !DISubroutineType(types: !29)
!29 = !{!7}
!30 = !{!31, !32}
!31 = !DILocalVariable(name: "l", scope: !27, file: !1, line: 27, type: !7)
!32 = !DILocalVariable(name: "k", scope: !27, file: !1, line: 27, type: !7)
!33 = !DILocation(line: 27, column: 3, scope: !27)
!34 = !DILocation(line: 0, scope: !27)
!35 = !DILocation(line: 29, column: 3, scope: !27)
!36 = !DILocation(line: 30, column: 3, scope: !27)
!37 = !DILocation(line: 32, column: 8, scope: !27)
