; RUN: llc -mtriple=bpfel -mcpu=v4 < %s | FileCheck %s

; Make sure that CO-RE relocations had been generated correctly for
; BPF_ST (store immediate) instructions and that
; BPFMISimplifyPatchable optimizations had been applied.
;
; Generated from the following source code:
;
;   #define __pai __attribute__((preserve_access_index))
;
;   struct foo {
;     unsigned char  b;
;     unsigned short h;
;     unsigned int   w;
;     unsigned long  d;
;   } __pai;
;
;   void bar(volatile struct foo *p) {
;     p->b = 1;
;     p->h = 2;
;     p->w = 3;
;     p->d = 4;
;   }
;
; Using the following command:
;
;   clang -g -O2 -S -emit-llvm -mcpu=v4 --target=bpfel test.c

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"

@"llvm.foo:0:0$0:0" = external global i64, !llvm.preserve.access.index !0 #0
@"llvm.foo:0:2$0:1" = external global i64, !llvm.preserve.access.index !0 #0
@"llvm.foo:0:4$0:2" = external global i64, !llvm.preserve.access.index !0 #0
@"llvm.foo:0:8$0:3" = external global i64, !llvm.preserve.access.index !0 #0

; Function Attrs: nofree nounwind
define dso_local void @bar(ptr noundef %p) local_unnamed_addr #1 !dbg !18 {
entry:
  call void @llvm.dbg.value(metadata ptr %p, metadata !24, metadata !DIExpression()), !dbg !25
  %0 = load i64, ptr @"llvm.foo:0:0$0:0", align 8
  %1 = getelementptr i8, ptr %p, i64 %0
  %2 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 0, ptr %1)
  store volatile i8 1, ptr %2, align 8, !dbg !26, !tbaa !27
  %3 = load i64, ptr @"llvm.foo:0:2$0:1", align 8
  %4 = getelementptr i8, ptr %p, i64 %3
  %5 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 1, ptr %4)
  store volatile i16 2, ptr %5, align 2, !dbg !34, !tbaa !35
  %6 = load i64, ptr @"llvm.foo:0:4$0:2", align 8
  %7 = getelementptr i8, ptr %p, i64 %6
  %8 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 2, ptr %7)
  store volatile i32 3, ptr %8, align 4, !dbg !36, !tbaa !37
  %9 = load i64, ptr @"llvm.foo:0:8$0:3", align 8
  %10 = getelementptr i8, ptr %p, i64 %9
  %11 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 3, ptr %10)
  store volatile i64 4, ptr %11, align 8, !dbg !38, !tbaa !39
  ret void, !dbg !40
}

; CHECK: [[L0:.Ltmp.*]]:
; CHECK:       *(u8 *)(r1 + 0) = 1
; CHECK: [[L2:.Ltmp.*]]:
; CHECK:       *(u16 *)(r1 + 2) = 2
; CHECK: [[L4:.Ltmp.*]]:
; CHECK:       *(u32 *)(r1 + 4) = 3
; CHECK: [[L6:.Ltmp.*]]:
; CHECK:       *(u64 *)(r1 + 8) = 4

; CHECK:       .section        .BTF
; ...
; CHECK:       .long   [[FOO:.*]]           # BTF_KIND_STRUCT(id = [[FOO_ID:.*]])
; ...
; CHECK:       .ascii  "foo"                # string offset=[[FOO]]
; CHECK:       .ascii  ".text"              # string offset=[[TEXT:.*]]
; CHECK:       .ascii  "0:0"                # string offset=[[S1:.*]]
; CHECK:       .ascii  "0:1"                # string offset=[[S2:.*]]
; CHECK:       .ascii  "0:2"                # string offset=[[S3:.*]]
; CHECK:       .ascii  "0:3"                # string offset=[[S4:.*]]

; CHECK:       .section        .BTF.ext
; ...
; CHECK:       .long   [[#]]                # FieldReloc
; CHECK-NEXT:  .long   [[TEXT]]             # Field reloc section string offset=[[TEXT]]
; CHECK-NEXT:  .long   [[#]]
; CHECK-NEXT:  .long   [[L0]]
; CHECK-NEXT:  .long   [[FOO_ID]]
; CHECK-NEXT:  .long   [[S1]]
; CHECK-NEXT:  .long   0
; CHECK-NEXT:  .long   [[L2]]
; CHECK-NEXT:  .long   [[FOO_ID]]
; CHECK-NEXT:  .long   [[S2]]
; CHECK-NEXT:  .long   0
; CHECK-NEXT:  .long   [[L4]]
; CHECK-NEXT:  .long   [[FOO_ID]]
; CHECK-NEXT:  .long   [[S3]]
; CHECK-NEXT:  .long   0
; CHECK-NEXT:  .long   [[L6]]
; CHECK-NEXT:  .long   [[FOO_ID]]
; CHECK-NEXT:  .long   [[S4]]
; CHECK-NEXT:  .long   0

; Function Attrs: nofree nosync nounwind memory(none)
declare ptr @llvm.bpf.passthrough.p0.p0(i32, ptr) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

attributes #0 = { "btf_ama" }
attributes #1 = { nofree nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="v4" }
attributes #2 = { nofree nosync nounwind memory(none) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!11}
!llvm.module.flags = !{!12, !13, !14, !15, !16}
!llvm.ident = !{!17}

!0 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "foo", file: !1, line: 3, size: 128, elements: !2)
!1 = !DIFile(filename: "some-file.c", directory: "/some/dir", checksumkind: CSK_MD5, checksum: "e5d03b4d39dfffadc6c607e956c37996")
!2 = !{!3, !5, !7, !9}
!3 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !0, file: !1, line: 4, baseType: !4, size: 8)
!4 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!5 = !DIDerivedType(tag: DW_TAG_member, name: "h", scope: !0, file: !1, line: 5, baseType: !6, size: 16, offset: 16)
!6 = !DIBasicType(name: "unsigned short", size: 16, encoding: DW_ATE_unsigned)
!7 = !DIDerivedType(tag: DW_TAG_member, name: "w", scope: !0, file: !1, line: 6, baseType: !8, size: 32, offset: 32)
!8 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !0, file: !1, line: 7, baseType: !10, size: 64, offset: 64)
!10 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!11 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 18.0.0 ...", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!12 = !{i32 7, !"Dwarf Version", i32 5}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{i32 7, !"frame-pointer", i32 2}
!16 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!17 = !{!"clang version 18.0.0 ..."}
!18 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 10, type: !19, scopeLine: 10, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !11, retainedNodes: !23)
!19 = !DISubroutineType(types: !20)
!20 = !{null, !21}
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !22, size: 64)
!22 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !0)
!23 = !{!24}
!24 = !DILocalVariable(name: "p", arg: 1, scope: !18, file: !1, line: 10, type: !21)
!25 = !DILocation(line: 0, scope: !18)
!26 = !DILocation(line: 11, column: 8, scope: !18)
!27 = !{!28, !29, i64 0}
!28 = !{!"foo", !29, i64 0, !31, i64 2, !32, i64 4, !33, i64 8}
!29 = !{!"omnipotent char", !30, i64 0}
!30 = !{!"Simple C/C++ TBAA"}
!31 = !{!"short", !29, i64 0}
!32 = !{!"int", !29, i64 0}
!33 = !{!"long", !29, i64 0}
!34 = !DILocation(line: 12, column: 8, scope: !18)
!35 = !{!28, !31, i64 2}
!36 = !DILocation(line: 13, column: 8, scope: !18)
!37 = !{!28, !32, i64 4}
!38 = !DILocation(line: 14, column: 8, scope: !18)
!39 = !{!28, !33, i64 8}
!40 = !DILocation(line: 15, column: 1, scope: !18)
