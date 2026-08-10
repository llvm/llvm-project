; RUN: llc -mtriple=bpf -mcpu=v2 < %s | FileCheck -check-prefixes=CHECK %s

; Test for machine register liveness update bug in
; BPFMISimplifyPatchable::processDstReg.
;
; Generated from the following source code:
;   struct t {
;     unsigned long a;
;   } __attribute__((preserve_access_index));
;
;   void foo(volatile struct t *t, volatile unsigned long *p) {
;     *p = t->a;
;     *p = t->a;
;   }
;
; Using the following command:
;   clang -g -O2 -S -emit-llvm --target=bpf t.c -o t.ll

@"llvm.t:0:0$0:0" = external global i64, !llvm.preserve.access.index !0 #0

; Function Attrs: nofree nounwind
define dso_local void @foo(ptr noundef %t, ptr noundef %p) local_unnamed_addr #1 !dbg !12 {
entry:
  call void @llvm.dbg.value(metadata ptr %t, metadata !20, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata ptr %p, metadata !21, metadata !DIExpression()), !dbg !22
  %0 = load i64, ptr @"llvm.t:0:0$0:0", align 8
  %1 = getelementptr i8, ptr %t, i64 %0
  %2 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 0, ptr %1)
  %3 = load volatile i64, ptr %2, align 8, !dbg !23, !tbaa !24
  store volatile i64 %3, ptr %p, align 8, !dbg !29, !tbaa !30
  %4 = load i64, ptr @"llvm.t:0:0$0:0", align 8
  %5 = getelementptr i8, ptr %t, i64 %4
  %6 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 1, ptr %5)
  %7 = load volatile i64, ptr %6, align 8, !dbg !31, !tbaa !24
  store volatile i64 %7, ptr %p, align 8, !dbg !32, !tbaa !30
  ret void, !dbg !33
}

; CHECK:      foo:
; CHECK:              prologue_end
; CHECK-NEXT: .Ltmp[[LABEL1:.*]]:
; CHECK-NEXT: .Ltmp
; CHECK-NEXT:         r[[#A:]] = *(u64 *)(r1 + 0)
; CHECK-NEXT:         .loc
; CHECK-NEXT: .Ltmp
; CHECK-NEXT:         *(u64 *)(r2 + 0) = r[[#A]]
; CHECK-NEXT:         .loc
; CHECK-NEXT: .Ltmp[[LABEL2:.*]]:
; CHECK-NEXT: .Ltmp
; CHECK-NEXT:         r[[#B:]] = *(u64 *)(r1 + 0)
; CHECK-NEXT: .Ltmp
; CHECK-NEXT: .Ltmp
; CHECK-NEXT:         .loc
; CHECK-NEXT: .Ltmp
; CHECK-NEXT:         *(u64 *)(r2 + 0) = r[[#B]]

; CHECK: .section .BTF
; CHECK: .long [[STR_T:.*]]  # BTF_KIND_STRUCT(id = [[ID:.*]])

; CHECK: .byte   116     # string offset=[[STR_T]]
; CHECK: .ascii  "0:0"   # string offset=[[STR_A:.*]]

; CHECK:     # FieldReloc
; CHECK:      .long .Ltmp[[LABEL1]]
; CHECK-NEXT: .long [[ID]]
; CHECK-NEXT: .long [[STR_A]]
; CHECK-NEXT: .long 0
; CHECK:      .long .Ltmp[[LABEL2]]
; CHECK-NEXT: .long [[ID]]
; CHECK-NEXT: .long [[STR_A]]
; CHECK-NEXT: .long 0

; Function Attrs: nofree nosync nounwind memory(none)
declare ptr @llvm.bpf.passthrough.p0.p0(i32, ptr) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

attributes #0 = { "btf_ama" }
attributes #1 = { nofree nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { nofree nosync nounwind memory(none) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!5}
!llvm.module.flags = !{!6, !7, !8, !9, !10}
!llvm.ident = !{!11}

!0 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t", file: !1, line: 1, size: 64, elements: !2)
!1 = !DIFile(filename: "some.file", directory: "/some/dir", checksumkind: CSK_MD5, checksum: "a149cfaf65a83125e7f2b2f47e5c7287")
!2 = !{!3}
!3 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !0, file: !1, line: 2, baseType: !4, size: 64)
!4 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!5 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 18.0.0 (/home/eddy/work/llvm-project/clang 3810f2eb4382d5e2090ce5cd47f45379cb453c35)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!6 = !{i32 7, !"Dwarf Version", i32 5}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{i32 7, !"frame-pointer", i32 2}
!10 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!11 = !{!"clang version 18.0.0 (/home/eddy/work/llvm-project/clang 3810f2eb4382d5e2090ce5cd47f45379cb453c35)"}
!12 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 4, type: !13, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !5, retainedNodes: !19)
!13 = !DISubroutineType(types: !14)
!14 = !{null, !15, !17}
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!16 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !0)
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 64)
!18 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !4)
!19 = !{!20, !21}
!20 = !DILocalVariable(name: "t", arg: 1, scope: !12, file: !1, line: 4, type: !15)
!21 = !DILocalVariable(name: "p", arg: 2, scope: !12, file: !1, line: 4, type: !17)
!22 = !DILocation(line: 0, scope: !12)
!23 = !DILocation(line: 5, column: 11, scope: !12)
!24 = !{!25, !26, i64 0}
!25 = !{!"t", !26, i64 0}
!26 = !{!"long", !27, i64 0}
!27 = !{!"omnipotent char", !28, i64 0}
!28 = !{!"Simple C/C++ TBAA"}
!29 = !DILocation(line: 5, column: 6, scope: !12)
!30 = !{!26, !26, i64 0}
!31 = !DILocation(line: 6, column: 11, scope: !12)
!32 = !DILocation(line: 6, column: 6, scope: !12)
!33 = !DILocation(line: 7, column: 1, scope: !12)
