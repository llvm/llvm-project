; RUN: llc -mtriple=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -mtriple=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
;
; Source:
;   void foo(const ptr);
;   int test() {
;     const char *str = "abcd";
;     const struct {
;       unsigned a[4];
;     } val = { .a = {2, 3, 4, 5} };
;     foo(str);
;     foo(&val);
;     return 0;
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm t.c

%struct.anon = type { [4 x i32] }

@.str = private unnamed_addr constant [5 x i8] c"abcd\00", align 1
@__const.test.val = private unnamed_addr constant %struct.anon { [4 x i32] [i32 2, i32 3, i32 4, i32 5] }, align 4

; Function Attrs: nounwind
define dso_local i32 @test() local_unnamed_addr #0 !dbg !7 {
entry:
  %val = alloca %struct.anon, align 4
  call void @llvm.dbg.value(metadata ptr @.str, metadata !12, metadata !DIExpression()), !dbg !25
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %val) #4, !dbg !26
  call void @llvm.dbg.declare(metadata ptr %val, metadata !16, metadata !DIExpression()), !dbg !27
  call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 4 dereferenceable(16) %val, ptr nonnull align 4 dereferenceable(16) @__const.test.val, i64 16, i1 false), !dbg !27
  tail call void @foo(ptr @.str) #4, !dbg !28
  call void @foo(ptr nonnull %val) #4, !dbg !29
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %val) #4, !dbg !30
  ret i32 0, !dbg !31
}

; the initial value of "str" is stored in section .rodata.str1.1
; the initial value of "val" is stored in section .rodata.cst16
; CHECK-NOT:   BTF_KIND_DATASEC

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #1

declare !dbg !32 dso_local void @foo(ptr) local_unnamed_addr #3

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { nounwind readnone speculatable willreturn }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0 (https://github.com/llvm/llvm-project.git 1e92cffe18a07c12042b57504dfa7fb709b833c8)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "/tmp/home/yhs/tmp")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git 1e92cffe18a07c12042b57504dfa7fb709b833c8)"}
!7 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12, !16}
!12 = !DILocalVariable(name: "str", scope: !7, file: !1, line: 3, type: !13)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!14 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !15)
!15 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!16 = !DILocalVariable(name: "val", scope: !7, file: !1, line: 6, type: !17)
!17 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !18)
!18 = distinct !DICompositeType(tag: DW_TAG_structure_type, scope: !7, file: !1, line: 4, size: 128, elements: !19)
!19 = !{!20}
!20 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !18, file: !1, line: 5, baseType: !21, size: 128)
!21 = !DICompositeType(tag: DW_TAG_array_type, baseType: !22, size: 128, elements: !23)
!22 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!23 = !{!24}
!24 = !DISubrange(count: 4)
!25 = !DILocation(line: 0, scope: !7)
!26 = !DILocation(line: 4, column: 3, scope: !7)
!27 = !DILocation(line: 6, column: 5, scope: !7)
!28 = !DILocation(line: 7, column: 3, scope: !7)
!29 = !DILocation(line: 8, column: 3, scope: !7)
!30 = !DILocation(line: 10, column: 1, scope: !7)
!31 = !DILocation(line: 9, column: 3, scope: !7)
!32 = !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!33 = !DISubroutineType(types: !34)
!34 = !{null, !35}
!35 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !36, size: 64)
!36 = !DIDerivedType(tag: DW_TAG_const_type, baseType: null)
