; RUN: llc -mtriple=bpf -mcpu=v2 < %s | FileCheck -check-prefixes=CHECK,V2 %s
; RUN: llc -mtriple=bpf -mcpu=v4 < %s | FileCheck -check-prefixes=CHECK,V4 %s

; Verify that BPFMISimplifyPatchable::checkADDrr correctly rewrites
; store instructions.
;
; Generated from the following source code:
;   struct t {
;     unsigned char  ub;
;     unsigned short uh;
;     unsigned int   uw;
;     unsigned long  ud;
;   } __attribute__((preserve_access_index));
;
;   void foo(volatile struct t *t) {
;     t->ub = 1;
;     t->uh = 2;
;     t->uw = 3;
;     t->ud = 4;
;   }
;
; Using the following command:
;   clang -g -O2 -S -emit-llvm --target=bpf t.c -o t.ll

@"llvm.t:0:0$0:0" = external global i64, !llvm.preserve.access.index !0 #0
@"llvm.t:0:2$0:1" = external global i64, !llvm.preserve.access.index !0 #0
@"llvm.t:0:4$0:2" = external global i64, !llvm.preserve.access.index !0 #0
@"llvm.t:0:8$0:3" = external global i64, !llvm.preserve.access.index !0 #0

; Function Attrs: nofree nounwind
define dso_local void @foo(ptr noundef %t, i64 noundef %v) local_unnamed_addr #1 !dbg !18 {
entry:
  call void @llvm.dbg.value(metadata ptr %t, metadata !24, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i64 %v, metadata !25, metadata !DIExpression()), !dbg !26
  %conv = trunc i64 %v to i8, !dbg !27
  %0 = load i64, ptr @"llvm.t:0:0$0:0", align 8
  %1 = getelementptr i8, ptr %t, i64 %0
  %2 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 0, ptr %1)
  store volatile i8 %conv, ptr %2, align 8, !dbg !28, !tbaa !29
  %conv1 = trunc i64 %v to i16, !dbg !36
  %3 = load i64, ptr @"llvm.t:0:2$0:1", align 8
  %4 = getelementptr i8, ptr %t, i64 %3
  %5 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 1, ptr %4)
  store volatile i16 %conv1, ptr %5, align 2, !dbg !37, !tbaa !38
  %conv2 = trunc i64 %v to i32, !dbg !39
  %6 = load i64, ptr @"llvm.t:0:4$0:2", align 8
  %7 = getelementptr i8, ptr %t, i64 %6
  %8 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 2, ptr %7)
  store volatile i32 %conv2, ptr %8, align 4, !dbg !40, !tbaa !41
  %9 = load i64, ptr @"llvm.t:0:8$0:3", align 8
  %10 = getelementptr i8, ptr %t, i64 %9
  %11 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 3, ptr %10)
  store volatile i64 %v, ptr %11, align 8, !dbg !42, !tbaa !43
  ret void, !dbg !44
}

; CHECK: foo:
; CHECK:      prologue_end
; CHECK-NEXT: .Ltmp[[LABEL_UB:.*]]:
; CHECK-NEXT: .Ltmp
; V2-NEXT:            *(u8 *)(r1 + 0) = r2
; V4-NEXT:            *(u8 *)(r1 + 0) = w2
; CHECK-NEXT:         .loc
; CHECK-NEXT: .Ltmp[[LABEL_UH:.*]]:
; CHECK-NEXT: .Ltmp
; V2-NEXT:            *(u16 *)(r1 + 2) = r2
; V4-NEXT:            *(u16 *)(r1 + 2) = w2
; CHECK-NEXT:         .loc
; CHECK-NEXT: .Ltmp[[LABEL_UW:.*]]:
; CHECK-NEXT: .Ltmp
; V2-NEXT:            *(u32 *)(r1 + 4) = r2
; V4-NEXT:            *(u32 *)(r1 + 4) = w2
; CHECK-NEXT:         .loc
; CHECK-NEXT: .Ltmp[[LABEL_UD:.*]]:
; CHECK-NEXT: .Ltmp
; CHECK-NEXT:         *(u64 *)(r1 + 8) = r2

; CHECK: .section .BTF
; CHECK: .long [[STR_T:.*]]  # BTF_KIND_STRUCT(id = [[ID:.*]])

; CHECK: .byte   116     # string offset=[[STR_T]]
; CHECK: .ascii  "0:0"   # string offset=[[STR_UB:.*]]
; CHECK: .ascii  "0:1"   # string offset=[[STR_UH:.*]]
; CHECK: .ascii  "0:2"   # string offset=[[STR_UW:.*]]
; CHECK: .ascii  "0:3"   # string offset=[[STR_UD:.*]]

; CHECK:     # FieldReloc
; CHECK:      .long .Ltmp[[LABEL_UB]]
; CHECK-NEXT: .long [[ID]]
; CHECK-NEXT: .long [[STR_UB]]
; CHECK-NEXT: .long 0
; CHECK:      .long .Ltmp[[LABEL_UH]]
; CHECK-NEXT: .long [[ID]]
; CHECK-NEXT: .long [[STR_UH]]
; CHECK-NEXT: .long 0
; CHECK:      .long .Ltmp[[LABEL_UW]]
; CHECK-NEXT: .long [[ID]]
; CHECK-NEXT: .long [[STR_UW]]
; CHECK-NEXT: .long 0
; CHECK:      .long .Ltmp[[LABEL_UD]]
; CHECK-NEXT: .long [[ID]]
; CHECK-NEXT: .long [[STR_UD]]
; CHECK-NEXT: .long 0

; Function Attrs: nofree nosync nounwind memory(none)
declare ptr @llvm.bpf.passthrough.p0.p0(i32, ptr) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

attributes #0 = { "btf_ama" }
attributes #1 = { nofree nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { nofree nosync nounwind memory(none) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!11}
!llvm.module.flags = !{!12, !13, !14, !15, !16}
!llvm.ident = !{!17}

!0 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t", file: !1, line: 1, size: 128, elements: !2)
!1 = !DIFile(filename: "some.file", directory: "/some/dir", checksumkind: CSK_MD5, checksum: "2067f770ab52f9042a61e5bf50a913bd")
!2 = !{!3, !5, !7, !9}
!3 = !DIDerivedType(tag: DW_TAG_member, name: "ub", scope: !0, file: !1, line: 2, baseType: !4, size: 8)
!4 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!5 = !DIDerivedType(tag: DW_TAG_member, name: "uh", scope: !0, file: !1, line: 3, baseType: !6, size: 16, offset: 16)
!6 = !DIBasicType(name: "unsigned short", size: 16, encoding: DW_ATE_unsigned)
!7 = !DIDerivedType(tag: DW_TAG_member, name: "uw", scope: !0, file: !1, line: 4, baseType: !8, size: 32, offset: 32)
!8 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "ud", scope: !0, file: !1, line: 5, baseType: !10, size: 64, offset: 64)
!10 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!11 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 18.0.0 (/home/eddy/work/llvm-project/clang 3810f2eb4382d5e2090ce5cd47f45379cb453c35)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!12 = !{i32 7, !"Dwarf Version", i32 5}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{i32 7, !"frame-pointer", i32 2}
!16 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!17 = !{!"clang version 18.0.0 (/home/eddy/work/llvm-project/clang 3810f2eb4382d5e2090ce5cd47f45379cb453c35)"}
!18 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 13, type: !19, scopeLine: 13, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !11, retainedNodes: !23)
!19 = !DISubroutineType(types: !20)
!20 = !{null, !21, !10}
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !22, size: 64)
!22 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !0)
!23 = !{!24, !25}
!24 = !DILocalVariable(name: "t", arg: 1, scope: !18, file: !1, line: 13, type: !21)
!25 = !DILocalVariable(name: "v", arg: 2, scope: !18, file: !1, line: 13, type: !10)
!26 = !DILocation(line: 0, scope: !18)
!27 = !DILocation(line: 14, column: 11, scope: !18)
!28 = !DILocation(line: 14, column: 9, scope: !18)
!29 = !{!30, !31, i64 0}
!30 = !{!"t", !31, i64 0, !33, i64 2, !34, i64 4, !35, i64 8}
!31 = !{!"omnipotent char", !32, i64 0}
!32 = !{!"Simple C/C++ TBAA"}
!33 = !{!"short", !31, i64 0}
!34 = !{!"int", !31, i64 0}
!35 = !{!"long", !31, i64 0}
!36 = !DILocation(line: 15, column: 11, scope: !18)
!37 = !DILocation(line: 15, column: 9, scope: !18)
!38 = !{!30, !33, i64 2}
!39 = !DILocation(line: 16, column: 11, scope: !18)
!40 = !DILocation(line: 16, column: 9, scope: !18)
!41 = !{!30, !34, i64 4}
!42 = !DILocation(line: 17, column: 9, scope: !18)
!43 = !{!30, !35, i64 8}
!44 = !DILocation(line: 18, column: 1, scope: !18)
