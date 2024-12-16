; RUN: llc -mtriple=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -mtriple=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
;
; Source:
;   #define __tag1 __attribute__((btf_type_tag("tag1")))
;   #define __tag2 __attribute__((btf_type_tag("tag2")))
;
;   struct foo;
;   struct map_value {
;           struct foo __tag2 __tag1 *ptr;
;   };
;   void func(struct map_value *);
;   void test(void)
;   {
;           struct map_value v = {};
;           func(&v);
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm test.c

%struct.map_value = type { ptr }
%struct.foo = type opaque

; Function Attrs: nounwind
define dso_local void @test() local_unnamed_addr #0 !dbg !7 {
entry:
  %v = alloca %struct.map_value, align 8
  %0 = bitcast ptr %v to ptr, !dbg !20
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %0) #4, !dbg !20
  call void @llvm.dbg.declare(metadata ptr %v, metadata !11, metadata !DIExpression()), !dbg !21
  %1 = bitcast ptr %v to ptr, !dbg !21
  store i64 0, ptr %1, align 8, !dbg !21
  call void @func(ptr noundef nonnull %v) #4, !dbg !22
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %0) #4, !dbg !23
  ret void, !dbg !23
}

; CHECK:             .long   0                               # BTF_KIND_FUNC_PROTO(id = 1)
; CHECK-NEXT:        .long   218103808                       # 0xd000000
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   1                               # BTF_KIND_FUNC(id = 2)
; CHECK-NEXT:        .long   201326593                       # 0xc000001
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   0                               # BTF_KIND_FUNC_PROTO(id = 3)
; CHECK-NEXT:        .long   218103809                       # 0xd000001
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   0                               # BTF_KIND_PTR(id = 4)
; CHECK-NEXT:        .long   33554432                        # 0x2000000
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   62                              # BTF_KIND_STRUCT(id = 5)
; CHECK-NEXT:        .long   67108865                        # 0x4000001
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   72
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   0                               # 0x0
; CHECK-NEXT:        .long   76                              # BTF_KIND_TYPE_TAG(id = 6)
; CHECK-NEXT:        .long   301989888                       # 0x12000000
; CHECK-NEXT:        .long   9
; CHECK-NEXT:        .long   81                              # BTF_KIND_TYPE_TAG(id = 7)
; CHECK-NEXT:        .long   301989888                       # 0x12000000
; CHECK-NEXT:        .long   6
; CHECK-NEXT:        .long   0                               # BTF_KIND_PTR(id = 8)
; CHECK-NEXT:        .long   33554432                        # 0x2000000
; CHECK-NEXT:        .long   7
; CHECK-NEXT:        .long   86                              # BTF_KIND_FWD(id = 9)
; CHECK-NEXT:        .long   117440512                       # 0x7000000
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   90                              # BTF_KIND_FUNC(id = 10)
; CHECK-NEXT:        .long   201326594                       # 0xc000002
; CHECK-NEXT:        .long   3

; CHECK:             .ascii  "test"                          # string offset=1
; CHECK:             .ascii  "map_value"                     # string offset=62
; CHECK:             .ascii  "ptr"                           # string offset=72
; CHECK:             .ascii  "tag2"                          # string offset=76
; CHECK:             .ascii  "tag1"                          # string offset=81
; CHECK:             .ascii  "foo"                           # string offset=86
; CHECK:             .ascii  "func"                          # string offset=90

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: mustprogress nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

declare !dbg !24 dso_local void @func(ptr noundef) local_unnamed_addr #3

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

attributes #0 = { nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { argmemonly mustprogress nofree nosync nounwind willreturn }
attributes #2 = { mustprogress nofree nosync nounwind readnone speculatable willreturn }
attributes #3 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 15.0.0 (https://github.com/llvm/llvm-project.git 25e8505f515bc9ef6c13527ffc4a902bae3a9071)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/llvm/btf_tag_type", checksumkind: CSK_MD5, checksum: "7735a89e98603fee29d352a8e9db5acb")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{!"clang version 15.0.0 (https://github.com/llvm/llvm-project.git 25e8505f515bc9ef6c13527ffc4a902bae3a9071)"}
!7 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 9, type: !8, scopeLine: 10, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{!11}
!11 = !DILocalVariable(name: "v", scope: !7, file: !1, line: 11, type: !12)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "map_value", file: !1, line: 5, size: 64, elements: !13)
!13 = !{!14}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "ptr", scope: !12, file: !1, line: 6, baseType: !15, size: 64)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64, annotations: !17)
!16 = !DICompositeType(tag: DW_TAG_structure_type, name: "foo", file: !1, line: 4, flags: DIFlagFwdDecl)
!17 = !{!18, !19}
!18 = !{!"btf_type_tag", !"tag2"}
!19 = !{!"btf_type_tag", !"tag1"}
!20 = !DILocation(line: 11, column: 9, scope: !7)
!21 = !DILocation(line: 11, column: 26, scope: !7)
!22 = !DILocation(line: 12, column: 9, scope: !7)
!23 = !DILocation(line: 13, column: 1, scope: !7)
!24 = !DISubprogram(name: "func", scope: !1, file: !1, line: 8, type: !25, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !28)
!25 = !DISubroutineType(types: !26)
!26 = !{null, !27}
!27 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!28 = !{}
