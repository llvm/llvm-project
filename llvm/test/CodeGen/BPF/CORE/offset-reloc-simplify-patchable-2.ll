; RUN: llc -mtriple=bpf -mcpu=v2 < %s | FileCheck -check-prefixes=CHECK,V2 %s
; RUN: llc -mtriple=bpf -mcpu=v4 < %s | FileCheck -check-prefixes=CHECK,V4 %s

; Verify that BPFMISimplifyPatchable::checkADDrr correctly rewrites
; load instructions.
;
; Generated from the following source code:
;   struct t {
;     unsigned char  ub;
;     unsigned short uh;
;     unsigned int   uw;
;     unsigned long  ud;
;   } __attribute__((preserve_access_index));
;
;   extern void cu(unsigned long);
;
;   void bar(volatile struct t *t) {
;     cu(t->ub);
;     cu(t->uh);
;     cu(t->uw);
;     cu(t->ud);
;   }
;
; Using the following command:
;   clang -g -O2 -S -emit-llvm --target=bpf t.c -o t.ll

@"llvm.t:0:0$0:0" = external global i64, !llvm.preserve.access.index !0 #0
@"llvm.t:0:2$0:1" = external global i64, !llvm.preserve.access.index !0 #0
@"llvm.t:0:4$0:2" = external global i64, !llvm.preserve.access.index !0 #0
@"llvm.t:0:8$0:3" = external global i64, !llvm.preserve.access.index !0 #0

; Function Attrs: nounwind
define dso_local void @bar(ptr noundef %t) local_unnamed_addr #1 !dbg !18 {
entry:
  call void @llvm.dbg.value(metadata ptr %t, metadata !24, metadata !DIExpression()), !dbg !25
  %0 = load i64, ptr @"llvm.t:0:0$0:0", align 8
  %1 = getelementptr i8, ptr %t, i64 %0
  %2 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 0, ptr %1)
  %3 = load volatile i8, ptr %2, align 8, !dbg !26, !tbaa !27
  %conv = zext i8 %3 to i64, !dbg !34
  tail call void @cu(i64 noundef %conv) #5, !dbg !35
  %4 = load i64, ptr @"llvm.t:0:2$0:1", align 8
  %5 = getelementptr i8, ptr %t, i64 %4
  %6 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 1, ptr %5)
  %7 = load volatile i16, ptr %6, align 2, !dbg !36, !tbaa !37
  %conv1 = zext i16 %7 to i64, !dbg !38
  tail call void @cu(i64 noundef %conv1) #5, !dbg !39
  %8 = load i64, ptr @"llvm.t:0:4$0:2", align 8
  %9 = getelementptr i8, ptr %t, i64 %8
  %10 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 2, ptr %9)
  %11 = load volatile i32, ptr %10, align 4, !dbg !40, !tbaa !41
  %conv2 = zext i32 %11 to i64, !dbg !42
  tail call void @cu(i64 noundef %conv2) #5, !dbg !43
  %12 = load i64, ptr @"llvm.t:0:8$0:3", align 8
  %13 = getelementptr i8, ptr %t, i64 %12
  %14 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 3, ptr %13)
  %15 = load volatile i64, ptr %14, align 8, !dbg !44, !tbaa !45
  tail call void @cu(i64 noundef %15) #5, !dbg !46
  ret void, !dbg !47
}

; CHECK: bar:
; CHECK:      prologue_end
; CHECK-NEXT: .Ltmp[[LABEL_UB:.*]]:
; CHECK-NEXT: .Ltmp
; V2-NEXT:        r1 = *(u8 *)(r6 + 0)
; V4-NEXT:        w1 = *(u8 *)(r6 + 0)
; CHECK-NEXT:     .loc
; CHECK-NEXT: .Ltmp
; CHECK-NEXT:     call cu
; CHECK-NEXT: .Ltmp
; CHECK-NEXT:     .loc
; CHECK-NEXT: .Ltmp[[LABEL_UH:.*]]:
; CHECK-NEXT: .Ltmp
; V2-NEXT:        r1 = *(u16 *)(r6 + 2)
; V4-NEXT:        w1 = *(u16 *)(r6 + 2)
; CHECK-NEXT:     .loc
; CHECK-NEXT: .Ltmp
; CHECK-NEXT:     call cu
; CHECK-NEXT: .Ltmp
; CHECK-NEXT:     .loc
; CHECK-NEXT: .Ltmp[[LABEL_UW:.*]]:
; CHECK-NEXT: .Ltmp
; V2-NEXT:        r1 = *(u32 *)(r6 + 4)
; V4-NEXT:        w1 = *(u32 *)(r6 + 4)
; CHECK-NEXT:     .loc
; CHECK-NEXT: .Ltmp
; CHECK-NEXT:     call cu
; CHECK-NEXT: .Ltmp
; CHECK-NEXT:     .loc
; CHECK-NEXT: .Ltmp[[LABEL_UD:.*]]:
; CHECK-NEXT: .Ltmp
; CHECK-NEXT:     r1 = *(u64 *)(r6 + 8)
; CHECK-NEXT:     .loc
; CHECK-NEXT: .Ltmp
; CHECK-NEXT:     call cu

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

declare !dbg !48 dso_local void @cu(i64 noundef) local_unnamed_addr #2

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

!llvm.dbg.cu = !{!11}
!llvm.module.flags = !{!12, !13, !14, !15, !16}
!llvm.ident = !{!17}

!0 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t", file: !1, line: 1, size: 128, elements: !2)
!1 = !DIFile(filename: "some.file", directory: "/some/dir", checksumkind: CSK_MD5, checksum: "d08c6eeba11118106c69a68932003da2")
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
!18 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 20, type: !19, scopeLine: 20, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !11, retainedNodes: !23)
!19 = !DISubroutineType(types: !20)
!20 = !{null, !21}
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !22, size: 64)
!22 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !0)
!23 = !{!24}
!24 = !DILocalVariable(name: "t", arg: 1, scope: !18, file: !1, line: 20, type: !21)
!25 = !DILocation(line: 0, scope: !18)
!26 = !DILocation(line: 21, column: 9, scope: !18)
!27 = !{!28, !29, i64 0}
!28 = !{!"t", !29, i64 0, !31, i64 2, !32, i64 4, !33, i64 8}
!29 = !{!"omnipotent char", !30, i64 0}
!30 = !{!"Simple C/C++ TBAA"}
!31 = !{!"short", !29, i64 0}
!32 = !{!"int", !29, i64 0}
!33 = !{!"long", !29, i64 0}
!34 = !DILocation(line: 21, column: 6, scope: !18)
!35 = !DILocation(line: 21, column: 3, scope: !18)
!36 = !DILocation(line: 22, column: 9, scope: !18)
!37 = !{!28, !31, i64 2}
!38 = !DILocation(line: 22, column: 6, scope: !18)
!39 = !DILocation(line: 22, column: 3, scope: !18)
!40 = !DILocation(line: 23, column: 9, scope: !18)
!41 = !{!28, !32, i64 4}
!42 = !DILocation(line: 23, column: 6, scope: !18)
!43 = !DILocation(line: 23, column: 3, scope: !18)
!44 = !DILocation(line: 24, column: 9, scope: !18)
!45 = !{!28, !33, i64 8}
!46 = !DILocation(line: 24, column: 3, scope: !18)
!47 = !DILocation(line: 25, column: 1, scope: !18)
!48 = !DISubprogram(name: "cu", scope: !1, file: !1, line: 10, type: !49, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !51)
!49 = !DISubroutineType(types: !50)
!50 = !{null, !10}
!51 = !{!52}
!52 = !DILocalVariable(arg: 1, scope: !48, file: !1, line: 10, type: !10)
