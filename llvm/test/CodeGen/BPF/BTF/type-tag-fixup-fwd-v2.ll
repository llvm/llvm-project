; RUN: llc -march=bpfel -filetype=obj -o - %s \
; RUN: | llvm-objcopy --dump-section .BTF=- - | %python %S/print_btf.py - | FileCheck %s
; RUN: llc -march=bpfeb -filetype=obj -o - %s \
; RUN: | llvm-objcopy --dump-section .BTF=- - | %python %S/print_btf.py - | FileCheck %s
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
;   clang -mllvm -btf-type-tag-v2 -target bpf -O2 -g -S -emit-llvm test.c

; Check generation of type tags attached to forward declaration.

; CHECK: [1] FUNC_PROTO '(anon)' ret_type_id=0 vlen=0
; CHECK: [2] FUNC 'test' type_id=1 linkage=global
; CHECK: [3] FUNC_PROTO '(anon)' ret_type_id=0 vlen=1
; CHECK: 	'(anon)' type_id=4
; CHECK: [4] PTR '(anon)' type_id=5
; CHECK: [5] STRUCT 'map_value' size=8 vlen=1
; CHECK: 	'ptr' type_id=6 bits_offset=0
; CHECK: [6] PTR '(anon)' type_id=9
; CHECK: [7] FWD 'foo' fwd_kind=struct
; CHECK: [8] TYPE_TAG 'tag2' type_id=7
; CHECK: [9] TYPE_TAG 'tag1' type_id=8
; CHECK: [10] FUNC 'func' type_id=3 linkage=extern

%struct.map_value = type { ptr }

; Function Attrs: nounwind
define dso_local void @test() local_unnamed_addr #0 !dbg !7 {
entry:
  %v = alloca %struct.map_value, align 8
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %v) #4, !dbg !23
  call void @llvm.dbg.declare(metadata ptr %v, metadata !11, metadata !DIExpression()), !dbg !24
  store i64 0, ptr %v, align 8, !dbg !24
  call void @func(ptr noundef nonnull %v) #4, !dbg !25
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %v) #4, !dbg !26
  ret void, !dbg !26
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

declare !dbg !27 dso_local void @func(ptr noundef) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

attributes #0 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang, some", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "some-file.c", directory: "/some/dir/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{!"clang, some version"}
!7 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 9, type: !8, scopeLine: 10, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{!11}
!11 = !DILocalVariable(name: "v", scope: !7, file: !1, line: 11, type: !12)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "map_value", file: !1, line: 5, size: 64, elements: !13)
!13 = !{!14}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "ptr", scope: !12, file: !1, line: 6, baseType: !15, size: 64)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!16 = !DICompositeType(tag: DW_TAG_structure_type, name: "foo", file: !1, line: 4, flags: DIFlagFwdDecl, annotations: !17)
!17 = !{!18, !19}
!18 = !{!"btf:type_tag", !"tag2"}
!19 = !{!"btf:type_tag", !"tag1"}
!23 = !DILocation(line: 11, column: 9, scope: !7)
!24 = !DILocation(line: 11, column: 26, scope: !7)
!25 = !DILocation(line: 12, column: 9, scope: !7)
!26 = !DILocation(line: 13, column: 1, scope: !7)
!27 = !DISubprogram(name: "func", scope: !1, file: !1, line: 8, type: !28, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !31)
!28 = !DISubroutineType(types: !29)
!29 = !{null, !30}
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!31 = !{!32}
!32 = !DILocalVariable(arg: 1, scope: !27, file: !1, line: 8, type: !30)
