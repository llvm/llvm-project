; RUN: llc -mtriple=bpf -mcpu=v4 < %s | FileCheck -check-prefixes=CHECK %s

; Verify that BPFMISimplifyPatchable::checkADDrr correctly rewrites
; load instructions with sign extension.
;
; Generated from the following source code:
;   struct t {
;     signed char    sb;
;     signed short   sh;
;     signed int     sw;
;   } __attribute__((preserve_access_index));
;
;   extern void cs(signed long);
;
;   void buz(volatile struct t *t) {
;     cs(t->sb);
;     cs(t->sh);
;     cs(t->sw);
;   }
;
; Using the following command:
;   clang -g -O2 -S -emit-llvm --target=bpf t.c -o t.ll

@"llvm.t:0:0$0:0" = external global i64, !llvm.preserve.access.index !0 #0
@"llvm.t:0:2$0:1" = external global i64, !llvm.preserve.access.index !0 #0
@"llvm.t:0:4$0:2" = external global i64, !llvm.preserve.access.index !0 #0

; Function Attrs: nounwind
define dso_local void @buz(ptr noundef %t) local_unnamed_addr #1 !dbg !16 {
entry:
  call void @llvm.dbg.value(metadata ptr %t, metadata !22, metadata !DIExpression()), !dbg !23
  %0 = load i64, ptr @"llvm.t:0:0$0:0", align 8
  %1 = getelementptr i8, ptr %t, i64 %0
  %2 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 0, ptr %1)
  %3 = load volatile i8, ptr %2, align 4, !dbg !24, !tbaa !25
  %conv = sext i8 %3 to i64, !dbg !31
  tail call void @cs(i64 noundef %conv) #5, !dbg !32
  %4 = load i64, ptr @"llvm.t:0:2$0:1", align 8
  %5 = getelementptr i8, ptr %t, i64 %4
  %6 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 1, ptr %5)
  %7 = load volatile i16, ptr %6, align 2, !dbg !33, !tbaa !34
  %conv1 = sext i16 %7 to i64, !dbg !35
  tail call void @cs(i64 noundef %conv1) #5, !dbg !36
  %8 = load i64, ptr @"llvm.t:0:4$0:2", align 8
  %9 = getelementptr i8, ptr %t, i64 %8
  %10 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 2, ptr %9)
  %11 = load volatile i32, ptr %10, align 4, !dbg !37, !tbaa !38
  %conv2 = sext i32 %11 to i64, !dbg !39
  tail call void @cs(i64 noundef %conv2) #5, !dbg !40
  ret void, !dbg !41
}

; CHECK: buz:
; CHECK:      prologue_end
; CHECK-NEXT: .Ltmp[[LABEL_SB:.*]]:
; CHECK-NEXT: .Ltmp
; CHECK-NEXT:     r1 = *(s8 *)(r6 + 0)
; CHECK-NEXT:     .loc
; CHECK-NEXT: .Ltmp
; CHECK-NEXT:     call cs
; CHECK-NEXT: .Ltmp
; CHECK-NEXT:     .loc
; CHECK-NEXT: .Ltmp[[LABEL_SH:.*]]:
; CHECK-NEXT: .Ltmp
; CHECK-NEXT:     r1 = *(s16 *)(r6 + 2)
; CHECK-NEXT:     .loc
; CHECK-NEXT: .Ltmp
; CHECK-NEXT:     call cs
; CHECK-NEXT: .Ltmp
; CHECK-NEXT:     .loc
; CHECK-NEXT: .Ltmp[[LABEL_SW:.*]]:
; CHECK-NEXT: .Ltmp
; CHECK-NEXT:     r1 = *(s32 *)(r6 + 4)
; CHECK-NEXT:     .loc
; CHECK-NEXT: .Ltmp
; CHECK-NEXT:     call cs

; CHECK: .section .BTF
; CHECK: .long [[T:.*]]  # BTF_KIND_STRUCT(id = [[ID:.*]])

; CHECK: .byte   116     # string offset=[[T]]
; CHECK: .ascii  "0:0"   # string offset=[[STR_SB:.*]]
; CHECK: .ascii  "0:1"   # string offset=[[STR_SH:.*]]
; CHECK: .ascii  "0:2"   # string offset=[[STR_SW:.*]]

; CHECK:     # FieldReloc
; CHECK:      .long .Ltmp[[LABEL_SB]]
; CHECK-NEXT: .long [[ID]]
; CHECK-NEXT: .long [[STR_SB]]
; CHECK-NEXT: .long 0
; CHECK:      .long .Ltmp[[LABEL_SH]]
; CHECK-NEXT: .long [[ID]]
; CHECK-NEXT: .long [[STR_SH]]
; CHECK-NEXT: .long 0
; CHECK:      .long .Ltmp[[LABEL_SW]]
; CHECK-NEXT: .long [[ID]]
; CHECK-NEXT: .long [[STR_SW]]
; CHECK-NEXT: .long 0

declare !dbg !42 dso_local void @cs(i64 noundef) local_unnamed_addr #2

; Function Attrs: nofree nosync nounwind memory(none)
declare ptr @llvm.bpf.passthrough.p0.p0(i32, ptr) #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #4

attributes #0 = { "btf_ama" }
attributes #1 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { nofree nosync nounwind memory(none) }
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { nounwind }

!llvm.dbg.cu = !{!9}
!llvm.module.flags = !{!10, !11, !12, !13, !14}
!llvm.ident = !{!15}

!0 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t", file: !1, line: 1, size: 64, elements: !2)
!1 = !DIFile(filename: "some.file", directory: "/some/dir", checksumkind: CSK_MD5, checksum: "2316ba0d3e8def5d297ad400e78b1782")
!2 = !{!3, !5, !7}
!3 = !DIDerivedType(tag: DW_TAG_member, name: "sb", scope: !0, file: !1, line: 6, baseType: !4, size: 8)
!4 = !DIBasicType(name: "signed char", size: 8, encoding: DW_ATE_signed_char)
!5 = !DIDerivedType(tag: DW_TAG_member, name: "sh", scope: !0, file: !1, line: 7, baseType: !6, size: 16, offset: 16)
!6 = !DIBasicType(name: "short", size: 16, encoding: DW_ATE_signed)
!7 = !DIDerivedType(tag: DW_TAG_member, name: "sw", scope: !0, file: !1, line: 8, baseType: !8, size: 32, offset: 32)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 18.0.0 (/home/eddy/work/llvm-project/clang 3810f2eb4382d5e2090ce5cd47f45379cb453c35)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!10 = !{i32 7, !"Dwarf Version", i32 5}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{i32 7, !"frame-pointer", i32 2}
!14 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!15 = !{!"clang version 18.0.0 (/home/eddy/work/llvm-project/clang 3810f2eb4382d5e2090ce5cd47f45379cb453c35)"}
!16 = distinct !DISubprogram(name: "buz", scope: !1, file: !1, line: 27, type: !17, scopeLine: 27, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !9, retainedNodes: !21)
!17 = !DISubroutineType(types: !18)
!18 = !{null, !19}
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !20, size: 64)
!20 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !0)
!21 = !{!22}
!22 = !DILocalVariable(name: "t", arg: 1, scope: !16, file: !1, line: 27, type: !19)
!23 = !DILocation(line: 0, scope: !16)
!24 = !DILocation(line: 28, column: 9, scope: !16)
!25 = !{!26, !27, i64 0}
!26 = !{!"t", !27, i64 0, !29, i64 2, !30, i64 4}
!27 = !{!"omnipotent char", !28, i64 0}
!28 = !{!"Simple C/C++ TBAA"}
!29 = !{!"short", !27, i64 0}
!30 = !{!"int", !27, i64 0}
!31 = !DILocation(line: 28, column: 6, scope: !16)
!32 = !DILocation(line: 28, column: 3, scope: !16)
!33 = !DILocation(line: 29, column: 9, scope: !16)
!34 = !{!26, !29, i64 2}
!35 = !DILocation(line: 29, column: 6, scope: !16)
!36 = !DILocation(line: 29, column: 3, scope: !16)
!37 = !DILocation(line: 30, column: 9, scope: !16)
!38 = !{!26, !30, i64 4}
!39 = !DILocation(line: 30, column: 6, scope: !16)
!40 = !DILocation(line: 30, column: 3, scope: !16)
!41 = !DILocation(line: 31, column: 1, scope: !16)
!42 = !DISubprogram(name: "cs", scope: !1, file: !1, line: 11, type: !43, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !46)
!43 = !DISubroutineType(types: !44)
!44 = !{null, !45}
!45 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!46 = !{!47}
!47 = !DILocalVariable(arg: 1, scope: !42, file: !1, line: 11, type: !45)
