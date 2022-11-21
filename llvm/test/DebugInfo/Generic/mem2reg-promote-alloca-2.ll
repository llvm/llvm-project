; RUN: opt -passes=mem2reg %s -S -o - | FileCheck %s

;; Check that mem2reg removes dbg.value(%local, DIExpression(DW_OP_deref...))
;; that instcombine LowerDbgDeclare inserted before the call to 'esc' when
;; promoting the alloca %local after 'esc' has been inlined. Without this we
;; provide no location for 'local', even though it is provably constant
;; throughout after inlining.
;;
;; $ clang reduce.c -O2 -g -emit-llvm -S -o tmp.ll -Xclang -disable-llvm-passes
;; $ opt tmp.ll -o - -instcombine -inline -S
;; $ cat reduce.c
;; long a;
;; int b;
;; void c();
;; __attribute__((__always_inline__))
;; static void esc(long *e) {
;;   *e = a;
;;   c();
;;   if (b)
;;     *e = 0;
;; }
;;
;; void fun() {
;;   long local = 0;
;;   esc(&local);
;; }

; CHECK: define dso_local void @fun()
; CHECK-NEXT: entry:
; CHECK-NEXT: call void @llvm.dbg.value(metadata i64 0, metadata ![[LOCAL:[0-9]+]], metadata !DIExpression())
; CHECK-NOT: call void @llvm.dbg.value({{.*}}, metadata ![[LOCAL]]
; CHECK: ![[LOCAL]] = !DILocalVariable(name: "local",

@a = dso_local global i64 0, align 8, !dbg !0
@b = dso_local global i32 0, align 4, !dbg !6

define dso_local void @fun() !dbg !14 {
entry:
  %e.addr.i = alloca i64*, align 8
  %local = alloca i64, align 8
  %0 = bitcast i64* %local to i8*, !dbg !19
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0), !dbg !19
  call void @llvm.dbg.value(metadata i64 0, metadata !18, metadata !DIExpression()), !dbg !20
  store i64 0, i64* %local, align 8, !dbg !21
  call void @llvm.dbg.value(metadata i64* %local, metadata !18, metadata !DIExpression(DW_OP_deref)), !dbg !20
  %1 = bitcast i64** %e.addr.i to i8*, !dbg !26
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %1), !dbg !26
  call void @llvm.dbg.value(metadata i64* %local, metadata !32, metadata !DIExpression()), !dbg !26
  store i64* %local, i64** %e.addr.i, align 8
  %2 = load i64, i64* @a, align 8, !dbg !36
  call void @llvm.dbg.value(metadata i64* %local, metadata !32, metadata !DIExpression()), !dbg !26
  store i64 %2, i64* %local, align 8, !dbg !37
  call void (...) @c(), !dbg !38
  %3 = load i32, i32* @b, align 4, !dbg !39
  %tobool.not.i = icmp eq i32 %3, 0, !dbg !39
  br i1 %tobool.not.i, label %esc.exit, label %if.then.i, !dbg !43

if.then.i:                                        ; preds = %entry
  %4 = load i64*, i64** %e.addr.i, align 8, !dbg !44
  call void @llvm.dbg.value(metadata i64* %4, metadata !32, metadata !DIExpression()), !dbg !26
  store i64 0, i64* %4, align 8, !dbg !45
  br label %esc.exit, !dbg !46

esc.exit:                                         ; preds = %entry, %if.then.i
  %5 = bitcast i64** %e.addr.i to i8*, !dbg !47
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %5), !dbg !47
  %6 = bitcast i64* %local to i8*, !dbg !48
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6), !dbg !48
  ret void, !dbg !48
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)
declare void @llvm.dbg.value(metadata, metadata, metadata)
declare !dbg !49 dso_local void @c(...)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11, !12}
!llvm.ident = !{!13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 1, type: !9, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 12.0.0)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "reduce.c", directory: "/")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "b", scope: !2, file: !3, line: 2, type: !8, isLocal: false, isDefinition: true)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!10 = !{i32 7, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{!"clang version 12.0.0"}
!14 = distinct !DISubprogram(name: "fun", scope: !3, file: !3, line: 12, type: !15, scopeLine: 12, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !17)
!15 = !DISubroutineType(types: !16)
!16 = !{null}
!17 = !{!18}
!18 = !DILocalVariable(name: "local", scope: !14, file: !3, line: 13, type: !9)
!19 = !DILocation(line: 13, column: 3, scope: !14)
!20 = !DILocation(line: 0, scope: !14)
!21 = !DILocation(line: 13, column: 8, scope: !14)
!26 = !DILocation(line: 0, scope: !27, inlinedAt: !33)
!27 = distinct !DISubprogram(name: "esc", scope: !3, file: !3, line: 5, type: !28, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !31)
!28 = !DISubroutineType(types: !29)
!29 = !{null, !30}
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!31 = !{!32}
!32 = !DILocalVariable(name: "e", arg: 1, scope: !27, file: !3, line: 5, type: !30)
!33 = distinct !DILocation(line: 14, column: 3, scope: !14)
!36 = !DILocation(line: 6, column: 8, scope: !27, inlinedAt: !33)
!37 = !DILocation(line: 6, column: 6, scope: !27, inlinedAt: !33)
!38 = !DILocation(line: 7, column: 3, scope: !27, inlinedAt: !33)
!39 = !DILocation(line: 8, column: 7, scope: !40, inlinedAt: !33)
!40 = distinct !DILexicalBlock(scope: !27, file: !3, line: 8, column: 7)
!43 = !DILocation(line: 8, column: 7, scope: !27, inlinedAt: !33)
!44 = !DILocation(line: 9, column: 6, scope: !40, inlinedAt: !33)
!45 = !DILocation(line: 9, column: 8, scope: !40, inlinedAt: !33)
!46 = !DILocation(line: 9, column: 5, scope: !40, inlinedAt: !33)
!47 = !DILocation(line: 10, column: 1, scope: !27, inlinedAt: !33)
!48 = !DILocation(line: 15, column: 1, scope: !14)
!49 = !DISubprogram(name: "c", scope: !3, file: !3, line: 3, type: !50, spFlags: DISPFlagOptimized, retainedNodes: !4)
!50 = !DISubroutineType(types: !51)
!51 = !{null, null}
