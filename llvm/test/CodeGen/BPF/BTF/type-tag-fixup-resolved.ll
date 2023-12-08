; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
;
; Source:
;   #define __tag1 __attribute__((btf_type_tag("tag1")))
;   #define __tag2 __attribute__((btf_type_tag("tag2")))
;
;   struct foo {
;           int i;
;   };
;   struct map_value {
;           struct foo __tag2 __tag1 *ptr;
;   };
;   void func(struct map_value *, struct foo *);
;   void test(void)
;   {
;           struct map_value v = {};
;           func(&v, v.ptr);
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm test.c

%struct.map_value = type { %struct.foo* }
%struct.foo = type { i32 }

; Function Attrs: nounwind
define dso_local void @test() local_unnamed_addr #0 !dbg !7 {
entry:
  %v = alloca %struct.map_value, align 8
  %0 = bitcast %struct.map_value* %v to i8*, !dbg !23
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #4, !dbg !23
  call void @llvm.dbg.declare(metadata %struct.map_value* %v, metadata !11, metadata !DIExpression()), !dbg !24
  %1 = bitcast %struct.map_value* %v to i64*, !dbg !24
  store i64 0, i64* %1, align 8, !dbg !24
  call void @func(%struct.map_value* noundef nonnull %v, %struct.foo* noundef null) #4, !dbg !25
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #4, !dbg !26
  ret void, !dbg !26
}

; CHECK:             .long   0                               # BTF_KIND_FUNC_PROTO(id = 1)
; CHECK-NEXT:        .long   218103808                       # 0xd000000
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   1                               # BTF_KIND_FUNC(id = 2)
; CHECK-NEXT:        .long   201326593                       # 0xc000001
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   0                               # BTF_KIND_FUNC_PROTO(id = 3)
; CHECK-NEXT:        .long   218103810                       # 0xd000002
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   7
; CHECK-NEXT:        .long   0                               # BTF_KIND_PTR(id = 4)
; CHECK-NEXT:        .long   33554432                        # 0x2000000
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   63                              # BTF_KIND_STRUCT(id = 5)
; CHECK-NEXT:        .long   67108865                        # 0x4000001
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   73
; CHECK-NEXT:        .long   6
; CHECK-NEXT:        .long   0                               # 0x0
; CHECK-NEXT:        .long   0                               # BTF_KIND_PTR(id = 6)
; CHECK-NEXT:        .long   33554432                        # 0x2000000
; CHECK-NEXT:        .long   12
; CHECK-NEXT:        .long   0                               # BTF_KIND_PTR(id = 7)
; CHECK-NEXT:        .long   33554432                        # 0x2000000
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   77                              # BTF_KIND_STRUCT(id = 8)
; CHECK-NEXT:        .long   67108865                        # 0x4000001
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   81
; CHECK-NEXT:        .long   9
; CHECK-NEXT:        .long   0                               # 0x0
; CHECK-NEXT:        .long   83                              # BTF_KIND_INT(id = 9)
; CHECK-NEXT:        .long   16777216                        # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                        # 0x1000020
; CHECK-NEXT:        .long   87                              # BTF_KIND_FUNC(id = 10)
; CHECK-NEXT:        .long   201326594                       # 0xc000002
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   92                              # BTF_KIND_TYPE_TAG(id = 11)
; CHECK-NEXT:        .long   301989888                       # 0x12000000
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   97                              # BTF_KIND_TYPE_TAG(id = 12)
; CHECK-NEXT:        .long   301989888                       # 0x12000000
; CHECK-NEXT:        .long   11

; CHECK:             .ascii  "test"                          # string offset=1
; CHECK:             .ascii  "map_value"                     # string offset=63
; CHECK:             .ascii  "ptr"                           # string offset=73
; CHECK:             .ascii  "foo"                           # string offset=77
; CHECK:             .byte   105                             # string offset=81
; CHECK:             .ascii  "int"                           # string offset=83
; CHECK:             .ascii  "func"                          # string offset=87
; CHECK:             .ascii  "tag2"                          # string offset=92
; CHECK:             .ascii  "tag1"                          # string offset=97

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: mustprogress nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

declare !dbg !27 dso_local void @func(%struct.map_value* noundef, %struct.foo* noundef) local_unnamed_addr #3

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

attributes #0 = { nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { argmemonly mustprogress nofree nosync nounwind willreturn }
attributes #2 = { mustprogress nofree nosync nounwind readnone speculatable willreturn }
attributes #3 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 15.0.0 (https://github.com/llvm/llvm-project.git 25e8505f515bc9ef6c13527ffc4a902bae3a9071)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp//home/yhs/work/tests/llvm/btf_tag_type", checksumkind: CSK_MD5, checksum: "8b3b8281c3b4240403467e0c9461251d")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{!"clang version 15.0.0 (https://github.com/llvm/llvm-project.git 25e8505f515bc9ef6c13527ffc4a902bae3a9071)"}
!7 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 11, type: !8, scopeLine: 12, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{!11}
!11 = !DILocalVariable(name: "v", scope: !7, file: !1, line: 13, type: !12)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "map_value", file: !1, line: 7, size: 64, elements: !13)
!13 = !{!14}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "ptr", scope: !12, file: !1, line: 8, baseType: !15, size: 64)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64, annotations: !20)
!16 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "foo", file: !1, line: 4, size: 32, elements: !17)
!17 = !{!18}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !16, file: !1, line: 5, baseType: !19, size: 32)
!19 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!20 = !{!21, !22}
!21 = !{!"btf_type_tag", !"tag2"}
!22 = !{!"btf_type_tag", !"tag1"}
!23 = !DILocation(line: 13, column: 9, scope: !7)
!24 = !DILocation(line: 13, column: 26, scope: !7)
!25 = !DILocation(line: 14, column: 9, scope: !7)
!26 = !DILocation(line: 15, column: 1, scope: !7)
!27 = !DISubprogram(name: "func", scope: !1, file: !1, line: 10, type: !28, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !32)
!28 = !DISubroutineType(types: !29)
!29 = !{null, !30, !31}
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!31 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!32 = !{}
