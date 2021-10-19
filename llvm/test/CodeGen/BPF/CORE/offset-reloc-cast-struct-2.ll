; RUN: opt -O2 %s | llvm-dis > %t1
; RUN: llc -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK %s
; RUN: llc -mattr=+alu32 -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK %s
; Source code:
;   struct v1 { int a; int b; };
;   typedef struct v1 __v1;
;   struct v2 { int c; int d; };
;   typedef struct v2 __v2;
;   struct v3 { char c; volatile const __v2 d; };
;   typedef struct v3 __v3;
;   #define _(x) (__builtin_preserve_access_index(x))
;   #define cast_to_v1(x) ((__v1 *)(x))
;   int get_value(const int *arg);
;   int test(__v3 *arg) {
;     return get_value(_(&cast_to_v1(&arg->d)->b));
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm -Xclang -disable-llvm-passes test.c

target triple = "bpf"

%struct.v3 = type { i8, %struct.v2 }
%struct.v2 = type { i32, i32 }
%struct.v1 = type { i32, i32 }

; Function Attrs: nounwind
define dso_local i32 @test(%struct.v3* %arg) local_unnamed_addr #0 !dbg !15 {
entry:
  call void @llvm.dbg.value(metadata %struct.v3* %arg, metadata !33, metadata !DIExpression()), !dbg !34
  %0 = tail call %struct.v2* @llvm.preserve.struct.access.index.p0s_struct.v2s.p0s_struct.v3s(%struct.v3* elementtype(%struct.v3) %arg, i32 1, i32 1), !dbg !35, !llvm.preserve.access.index !20
  %1 = bitcast %struct.v2* %0 to %struct.v1*, !dbg !35
  %2 = tail call i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.v1s(%struct.v1* elementtype(%struct.v1) %1, i32 1, i32 1), !dbg !35, !llvm.preserve.access.index !6
  %call = tail call i32 @get_value(i32* %2) #4, !dbg !36
  ret i32 %call, !dbg !37
}

; CHECK:             r2 = 4
; CHECK:             r1 += r2
; CHECK:             r2 = 4
; CHECK:             r1 += r2
; CHECK:             call get_value

; CHECK:             .long   6                       # BTF_KIND_STRUCT(id = [[TID1:[0-9]+]])
; CHECK:             .long   91                      # BTF_KIND_STRUCT(id = [[TID2:[0-9]+]])

; CHECK:             .ascii  "v3"                    # string offset=6
; CHECK:             .ascii  ".text"                 # string offset=39
; CHECK:             .ascii  "0:1"                   # string offset=45
; CHECK:             .ascii  "v1"                    # string offset=91


; CHECK:             .long   16                      # FieldReloc
; CHECK-NEXT:        .long   39                      # Field reloc section string offset=39
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   [[TID1]]
; CHECK-NEXT:        .long   45
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   [[TID2]]
; CHECK-NEXT:        .long   45
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
!llvm.module.flags = !{!11, !12, !13}
!llvm.ident = !{!14}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 (trunk 367256) (llvm/trunk 367266)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/llvm/cast")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!5 = !DIDerivedType(tag: DW_TAG_typedef, name: "__v1", file: !1, line: 2, baseType: !6)
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "v1", file: !1, line: 1, size: 64, elements: !7)
!7 = !{!8, !10}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !6, file: !1, line: 1, baseType: !9, size: 32)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !6, file: !1, line: 1, baseType: !9, size: 32, offset: 32)
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{!"clang version 10.0.0 (trunk 367256) (llvm/trunk 367266)"}
!15 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 10, type: !16, scopeLine: 10, flags: DIFlagPrototyped, isDefinition: true, isOptimized: true, unit: !0, retainedNodes: !32)
!16 = !DISubroutineType(types: !17)
!17 = !{!9, !18}
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !19, size: 64)
!19 = !DIDerivedType(tag: DW_TAG_typedef, name: "__v3", file: !1, line: 6, baseType: !20)
!20 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "v3", file: !1, line: 5, size: 96, elements: !21)
!21 = !{!22, !24}
!22 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !20, file: !1, line: 5, baseType: !23, size: 8)
!23 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!24 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !20, file: !1, line: 5, baseType: !25, size: 64, offset: 32)
!25 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !26)
!26 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !27)
!27 = !DIDerivedType(tag: DW_TAG_typedef, name: "__v2", file: !1, line: 4, baseType: !28)
!28 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "v2", file: !1, line: 3, size: 64, elements: !29)
!29 = !{!30, !31}
!30 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !28, file: !1, line: 3, baseType: !9, size: 32)
!31 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !28, file: !1, line: 3, baseType: !9, size: 32, offset: 32)
!32 = !{!33}
!33 = !DILocalVariable(name: "arg", arg: 1, scope: !15, file: !1, line: 10, type: !18)
!34 = !DILocation(line: 0, scope: !15)
!35 = !DILocation(line: 11, column: 20, scope: !15)
!36 = !DILocation(line: 11, column: 10, scope: !15)
!37 = !DILocation(line: 11, column: 3, scope: !15)
