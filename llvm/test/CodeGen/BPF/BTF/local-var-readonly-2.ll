; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
;
; Source:
;   void foo(const ptr);
;   int test() {
;     const struct {
;       unsigned a[4];
;       char b;
;     } val = { .a = {2, 3, 4, 5}, .b = 4 };
;     foo(&val);
;     return 0;
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm t.c

%struct.anon = type { [4 x i32], i8 }

@__const.test.val = private unnamed_addr constant %struct.anon { [4 x i32] [i32 2, i32 3, i32 4, i32 5], i8 4 }, align 4

; Function Attrs: nounwind
define dso_local i32 @test() local_unnamed_addr #0 !dbg !7 {
entry:
  %val = alloca %struct.anon, align 4
  call void @llvm.lifetime.start.p0(i64 20, ptr nonnull %val) #4, !dbg !23
  call void @llvm.dbg.declare(metadata ptr %val, metadata !12, metadata !DIExpression()), !dbg !24
  call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 4 dereferenceable(20) %val, ptr nonnull align 4 dereferenceable(20) @__const.test.val, i64 20, i1 false), !dbg !24
  call void @foo(ptr nonnull %val) #4, !dbg !25
  call void @llvm.lifetime.end.p0(i64 20, ptr nonnull %val) #4, !dbg !26
  ret i32 0, !dbg !27
}

; the init value of local variable "val" is stored in .rodata section
; CHECK:             .long   42                              # BTF_KIND_DATASEC
; CHECK-NEXT:        .long   251658240                       # 0xf000000
; CHECK-NEXT:        .long   0

; CHECK:             .ascii  ".rodata"                       # string offset=42

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #1

declare !dbg !28 dso_local void @foo(ptr) local_unnamed_addr #3

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

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
!11 = !{!12}
!12 = !DILocalVariable(name: "val", scope: !7, file: !1, line: 6, type: !13)
!13 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !14)
!14 = distinct !DICompositeType(tag: DW_TAG_structure_type, scope: !7, file: !1, line: 3, size: 160, elements: !15)
!15 = !{!16, !21}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !14, file: !1, line: 4, baseType: !17, size: 128)
!17 = !DICompositeType(tag: DW_TAG_array_type, baseType: !18, size: 128, elements: !19)
!18 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!19 = !{!20}
!20 = !DISubrange(count: 4)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !14, file: !1, line: 5, baseType: !22, size: 8, offset: 128)
!22 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!23 = !DILocation(line: 3, column: 3, scope: !7)
!24 = !DILocation(line: 6, column: 5, scope: !7)
!25 = !DILocation(line: 7, column: 3, scope: !7)
!26 = !DILocation(line: 9, column: 1, scope: !7)
!27 = !DILocation(line: 8, column: 3, scope: !7)
!28 = !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !29, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!29 = !DISubroutineType(types: !30)
!30 = !{null, !31}
!31 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !32, size: 64)
!32 = !DIDerivedType(tag: DW_TAG_const_type, baseType: null)
