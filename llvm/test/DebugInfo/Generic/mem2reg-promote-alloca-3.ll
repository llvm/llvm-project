; RUN: opt -passes=mem2reg %s -S -o - | FileCheck %s
; RUN: opt -passes=mem2reg %s -S -o - --try-experimental-debuginfo-iterators | FileCheck %s

;; Check that mem2reg removes dbg.value(%local, DIExpression(DW_OP_deref...))
;; that instcombine LowerDbgDeclare inserted before the call to 'esc' when
;; promoting the alloca %local after 'esc' has been inlined. Without this we
;; provide no location for 'local', even though it is provably constant
;; throughout after inlining.
;;
;; $ clang reduce.c -O2 -g -emit-llvm -S -o tmp.ll -Xclang -disable-llvm-passes
;; $ opt tmp.ll -o - -instcombine -inline -S
;; $ cat reduce.c
;; __attribute__((__always_inline__))
;; static void esc(unsigned char **c) {
;;   *c += 4;
;; }
;; void fun() {
;;   unsigned char *local = 0;
;;   esc(&local);
;; }

; CHECK: define dso_local void @fun()
; CHECK-NEXT: entry:
; CHECK-NEXT: call void @llvm.dbg.value(metadata ptr null, metadata ![[LOCAL:[0-9]+]], metadata !DIExpression())
; CHECK-NOT: call void @llvm.dbg.value({{.*}}, metadata ![[LOCAL]]
; CHECK: ![[LOCAL]] = !DILocalVariable(name: "local",

define dso_local void @fun() !dbg !7 {
entry:
  %local = alloca ptr, align 8
  %0 = bitcast ptr %local to ptr, !dbg !14
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %0) #3, !dbg !14
  call void @llvm.dbg.value(metadata ptr null, metadata !11, metadata !DIExpression()), !dbg !15
  store ptr null, ptr %local, align 8, !dbg !16
  call void @llvm.dbg.value(metadata ptr %local, metadata !11, metadata !DIExpression(DW_OP_deref)), !dbg !15
  call void @llvm.dbg.value(metadata ptr %local, metadata !21, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata ptr %local, metadata !21, metadata !DIExpression()), !dbg !27
  %1 = load ptr, ptr %local, align 8, !dbg !29
  %add.ptr.i = getelementptr inbounds i8, ptr %1, i64 4, !dbg !29
  store ptr %add.ptr.i, ptr %local, align 8, !dbg !29
  %2 = bitcast ptr %local to ptr, !dbg !30
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %2) #3, !dbg !30
  ret void, !dbg !30
}

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "reduce.c", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 12.0.0"}
!7 = distinct !DISubprogram(name: "fun", scope: !1, file: !1, line: 5, type: !8, scopeLine: 5, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{!11}
!11 = !DILocalVariable(name: "local", scope: !7, file: !1, line: 6, type: !12)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!13 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!14 = !DILocation(line: 6, column: 3, scope: !7)
!15 = !DILocation(line: 0, scope: !7)
!16 = !DILocation(line: 6, column: 18, scope: !7)
!21 = !DILocalVariable(name: "c", arg: 1, scope: !22, file: !1, line: 2, type: !25)
!22 = distinct !DISubprogram(name: "esc", scope: !1, file: !1, line: 2, type: !23, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !26)
!23 = !DISubroutineType(types: !24)
!24 = !{null, !25}
!25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!26 = !{!21}
!27 = !DILocation(line: 0, scope: !22, inlinedAt: !28)
!28 = distinct !DILocation(line: 7, column: 3, scope: !7)
!29 = !DILocation(line: 3, column: 6, scope: !22, inlinedAt: !28)
!30 = !DILocation(line: 8, column: 1, scope: !7)
