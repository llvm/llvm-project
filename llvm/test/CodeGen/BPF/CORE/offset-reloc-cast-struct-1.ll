; RUN: opt -O2 %s | llvm-dis > %t1
; RUN: llc -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK %s
; RUN: llc -mattr=+alu32 -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK %s
; Source code:
;   struct v1 { int a; int b; };
;   struct v2 { int c; int d; };
;   struct v3 { char c; struct v2 d; };
;   #define _(x) (__builtin_preserve_access_index(x))
;   #define cast_to_v1(x) ((struct v1 *)(x))
;   int get_value(const int *arg);
;   int test(struct v3 *arg) {
;     return get_value(_(&cast_to_v1(&arg->d)->b));
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm -Xclang -disable-llvm-passes test.c

target triple = "bpf"

%struct.v3 = type { i8, %struct.v2 }
%struct.v2 = type { i32, i32 }
%struct.v1 = type { i32, i32 }

; Function Attrs: nounwind
define dso_local i32 @test(%struct.v3* %arg) local_unnamed_addr #0 !dbg !14 {
entry:
  call void @llvm.dbg.value(metadata %struct.v3* %arg, metadata !28, metadata !DIExpression()), !dbg !29
  %0 = tail call %struct.v2* @llvm.preserve.struct.access.index.p0s_struct.v2s.p0s_struct.v3s(%struct.v3* elementtype(%struct.v3) %arg, i32 1, i32 1), !dbg !30, !llvm.preserve.access.index !18
  %1 = bitcast %struct.v2* %0 to %struct.v1*, !dbg !30
  %2 = tail call i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.v1s(%struct.v1* elementtype(%struct.v1) %1, i32 1, i32 1), !dbg !30, !llvm.preserve.access.index !5
  %call = tail call i32 @get_value(i32* %2) #4, !dbg !31
  ret i32 %call, !dbg !32
}

; CHECK:              r2 = 4
; CHECK:              r1 += r2
; CHECK:              r2 = 4
; CHECK:              r1 += r2
; CHECK:              call get_value

; CHECK:             .long   1                       # BTF_KIND_STRUCT(id = [[V3_TID:[0-9]+]])
; CHECK:             .long   81                      # BTF_KIND_STRUCT(id = [[V1_TID:[0-9]+]])

; CHECK:             .ascii  "v3"                    # string offset=1
; CHECK-NEXT:        .byte   0
; CHECK:             .ascii  ".text"                 # string offset=[[SEC_STR:[0-9]+]]
; CHECK-NEXT:        .byte   0
; CHECK:             .ascii  "0:1"                   # string offset=[[ACCESS_STR:[0-9]+]]
; CHECK-NEXT:        .byte   0
; CHECK:             .ascii  "v1"                    # string offset=81
; CHECK-NEXT:        .byte   0

; CHECK:             .long   16                      # FieldReloc
; CHECK-NEXT:        .long   [[SEC_STR]]             # Field reloc section string offset=[[SEC_STR]]
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   [[V3_TID]]
; CHECK-NEXT:        .long   [[ACCESS_STR]]
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   [[V1_TID]]
; CHECK-NEXT:        .long   [[ACCESS_STR]]
; CHECK-NEXT:        .long   0

declare dso_local i32 @get_value(i32*) local_unnamed_addr #1

; Function Attrs: nounwind readnone
declare %struct.v2* @llvm.preserve.struct.access.index.p0s_struct.v2s.p0s_struct.v3s(%struct.v3*, i32, i32) #2

; Function Attrs: nounwind readnone
declare i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.v1s(%struct.v1*, i32, i32) #2

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind readnone speculatable willreturn }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11, !12}
!llvm.ident = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 (trunk 367256) (llvm/trunk 367266)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/llvm/cast")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "v1", file: !1, line: 1, size: 64, elements: !6)
!6 = !{!7, !9}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !5, file: !1, line: 1, baseType: !8, size: 32)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !5, file: !1, line: 1, baseType: !8, size: 32, offset: 32)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{!"clang version 10.0.0 (trunk 367256) (llvm/trunk 367266)"}
!14 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 7, type: !15, scopeLine: 7, flags: DIFlagPrototyped, isDefinition: true, isOptimized: true, unit: !0, retainedNodes: !27)
!15 = !DISubroutineType(types: !16)
!16 = !{!8, !17}
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 64)
!18 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "v3", file: !1, line: 3, size: 96, elements: !19)
!19 = !{!20, !22}
!20 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !18, file: !1, line: 3, baseType: !21, size: 8)
!21 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!22 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !18, file: !1, line: 3, baseType: !23, size: 64, offset: 32)
!23 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "v2", file: !1, line: 2, size: 64, elements: !24)
!24 = !{!25, !26}
!25 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !23, file: !1, line: 2, baseType: !8, size: 32)
!26 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !23, file: !1, line: 2, baseType: !8, size: 32, offset: 32)
!27 = !{!28}
!28 = !DILocalVariable(name: "arg", arg: 1, scope: !14, file: !1, line: 7, type: !17)
!29 = !DILocation(line: 0, scope: !14)
!30 = !DILocation(line: 8, column: 20, scope: !14)
!31 = !DILocation(line: 8, column: 10, scope: !14)
!32 = !DILocation(line: 8, column: 3, scope: !14)
