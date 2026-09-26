; RUN: opt -O2 %s -S | FileCheck %s
; RUN: opt -O2 %s | llc -mtriple=bpfel -filetype=asm -o - | FileCheck %s --check-prefix=BTF

; Clang numbers the fields of a record for a CO-RE access, while the debug
; info of a C++ record also lists base classes, static data members, methods
; and the vtable pointer among its elements. The access index is mapped to the
; field it counts, and the access string index is the position of that field
; among the BTF members, where non-virtual bases come first as anonymous
; members and the vtable pointer is a member too. The name of each relocation
; global encodes <kind>:<patched value>$<access string>.
;
; struct Base { int b; };
; struct Base2 { int c; };
; struct S : Base, Base2 {
;   static int s1; int x; void m(); static int s2; int y;
; };
; struct V : virtual Base { int v; };   // vptr at 0, v at 8, Base at 12
; struct Empty {};
; struct E : Empty { int e; };          // Empty and e both at offset 0
; union U { int a; static int s; long b; void m(); };
; struct BF : Base {
;   void m(); unsigned x : 3; int y : 5; static int s; int z;
; };
;
; &s->x, &s->y, &v->v, &e->e, &u->b, and field info of bf->x, bf->y, bf->z

; CHECK-DAG: @"llvm.S:0:8$0:2" =
; CHECK-DAG: @"llvm.S:0:12$0:3" =
; CHECK-DAG: @"llvm.V:0:8$0:1" =
; CHECK-DAG: @"llvm.E:0:0$0:1" =
; CHECK-DAG: @"llvm.U:0:0$0:1" =
; CHECK-DAG: @"llvm.BF:1:4$0:2" =
; CHECK-DAG: @"llvm.BF:3:1$0:2" =
; CHECK-DAG: @"llvm.BF:3:0$0:1" =
; CHECK-DAG: @"llvm.BF:4:56$0:2" =
; CHECK-DAG: @"llvm.BF:0:8$0:3" =

; The relocation for &s->y names member 3 of the BTF struct S, which is y.
; S is the struct whose name is at string offset 1, and 83 is 'S'.
; BTF:      .long   1                               # BTF_KIND_STRUCT(id = [[S_ID:[0-9]+]])
; BTF:      .byte   83                              # string offset=1
; BTF:      .ascii  "0:3"                           # string offset=[[Y_ACCESS:[0-9]+]]
; BTF:      .section        .BTF.ext
; BTF:      .long   [[S_ID]]{{\n}}{{[[:space:]]+}}.long   [[Y_ACCESS]]{{\n}}{{[[:space:]]+}}.long   0

target triple = "bpf"

%struct.Base = type { i32 }
%struct.Base2 = type { i32 }
%struct.S = type { %struct.Base, %struct.Base2, i32, i32 }
%struct.V = type <{ ptr, i32, %struct.Base, [4 x i8] }>
%struct.E = type { i32 }
%struct.BF = type { %struct.Base, i8, i32 }

define ptr @s_x(ptr %p) {
  %r = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.S) %p, i32 2, i32 0), !llvm.preserve.access.index !10
  ret ptr %r
}

define ptr @s_y(ptr %p) !dbg !61 {
  %r = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.S) %p, i32 3, i32 1), !dbg !64, !llvm.preserve.access.index !10
  ret ptr %r
}

define ptr @v_v(ptr %p) {
  %r = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.V) %p, i32 1, i32 0), !llvm.preserve.access.index !20
  ret ptr %r
}

define ptr @e_e(ptr %p) {
  %r = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.E) %p, i32 0, i32 0), !llvm.preserve.access.index !30
  ret ptr %r
}

define ptr @u_b(ptr %p) {
  %r = call ptr @llvm.preserve.union.access.index.p0.p0(ptr %p, i32 1), !llvm.preserve.access.index !40
  ret ptr %r
}

define i32 @bf_y_size(ptr %p) {
  %f = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.BF) %p, i32 1, i32 1), !llvm.preserve.access.index !50
  %r = call i32 @llvm.bpf.preserve.field.info.p0(ptr %f, i64 1)
  ret i32 %r
}

define i32 @bf_y_signed(ptr %p) {
  %f = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.BF) %p, i32 1, i32 1), !llvm.preserve.access.index !50
  %r = call i32 @llvm.bpf.preserve.field.info.p0(ptr %f, i64 3)
  ret i32 %r
}

define i32 @bf_x_signed(ptr %p) {
  %f = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.BF) %p, i32 1, i32 0), !llvm.preserve.access.index !50
  %r = call i32 @llvm.bpf.preserve.field.info.p0(ptr %f, i64 3)
  ret i32 %r
}

define i32 @bf_y_lshift(ptr %p) {
  %f = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.BF) %p, i32 1, i32 1), !llvm.preserve.access.index !50
  %r = call i32 @llvm.bpf.preserve.field.info.p0(ptr %f, i64 4)
  ret i32 %r
}

define i32 @bf_z_offset(ptr %p) {
  %f = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.BF) %p, i32 2, i32 2), !llvm.preserve.access.index !50
  %r = call i32 @llvm.bpf.preserve.field.info.p0(ptr %f, i64 0)
  ret i32 %r
}

declare ptr @llvm.preserve.struct.access.index.p0.p0(ptr, i32 immarg, i32 immarg)
declare ptr @llvm.preserve.union.access.index.p0.p0(ptr, i32 immarg)
declare i32 @llvm.bpf.preserve.field.info.p0(ptr, i64 immarg)

!llvm.dbg.cu = !{!60}
!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!3 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!4 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Base", file: !1, size: 32, elements: !6)
!6 = !{!7}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !5, file: !1, baseType: !2, size: 32)
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Base2", file: !1, size: 32, elements: !9)
!9 = !{!17}
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !1, size: 128, elements: !11)
!11 = !{!12, !13, !14, !15, !16, !18, !19}
!12 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !10, baseType: !5, extraData: i32 0)
!13 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !10, baseType: !8, offset: 32, extraData: i32 0)
!14 = !DIDerivedType(tag: DW_TAG_variable, name: "s1", scope: !10, file: !1, baseType: !2, flags: DIFlagStaticMember)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !10, file: !1, baseType: !2, size: 32, offset: 64)
!16 = !DIDerivedType(tag: DW_TAG_variable, name: "s2", scope: !10, file: !1, baseType: !2, flags: DIFlagStaticMember)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !8, file: !1, baseType: !2, size: 32)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !10, file: !1, baseType: !2, size: 32, offset: 96)
!19 = !DISubprogram(name: "m", scope: !10, file: !1, type: !28, spFlags: 0)
!20 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "V", file: !1, size: 192, elements: !21)
!21 = !{!22, !23, !25}
!22 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !20, baseType: !5, offset: 24, flags: DIFlagVirtual, extraData: i32 0)
!23 = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$V", scope: !1, file: !1, baseType: !24, size: 64, flags: DIFlagArtificial)
!24 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!25 = !DIDerivedType(tag: DW_TAG_member, name: "v", scope: !20, file: !1, baseType: !2, size: 32, offset: 64)
!26 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Empty", file: !1, size: 8, elements: !27)
!27 = !{}
!28 = !DISubroutineType(types: !29)
!29 = !{null}
!30 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "E", file: !1, size: 32, elements: !31)
!31 = !{!32, !33}
!32 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !30, baseType: !26, extraData: i32 0)
!33 = !DIDerivedType(tag: DW_TAG_member, name: "e", scope: !30, file: !1, baseType: !2, size: 32)
!40 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "U", file: !1, size: 64, elements: !41)
!41 = !{!42, !43, !44, !45}
!42 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !40, file: !1, baseType: !2, size: 32)
!43 = !DIDerivedType(tag: DW_TAG_variable, name: "s", scope: !40, file: !1, baseType: !2, flags: DIFlagStaticMember)
!44 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !40, file: !1, baseType: !4, size: 64)
!45 = !DISubprogram(name: "m", scope: !40, file: !1, type: !28, spFlags: 0)
!50 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "BF", file: !1, size: 96, elements: !51)
!51 = !{!52, !53, !54, !55, !56, !57}
!52 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !50, baseType: !5, extraData: i32 0)
!53 = !DISubprogram(name: "m", scope: !50, file: !1, type: !28, spFlags: 0)
!54 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !50, file: !1, baseType: !3, size: 3, offset: 32, flags: DIFlagBitField, extraData: i64 32)
!55 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !50, file: !1, baseType: !2, size: 5, offset: 35, flags: DIFlagBitField, extraData: i64 32)
!56 = !DIDerivedType(tag: DW_TAG_variable, name: "s", scope: !50, file: !1, baseType: !2, flags: DIFlagStaticMember)
!57 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !50, file: !1, baseType: !2, size: 32, offset: 64)
!60 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!61 = distinct !DISubprogram(name: "s_y", scope: !1, file: !1, type: !62, spFlags: DISPFlagDefinition, unit: !60)
!62 = !DISubroutineType(types: !63)
!63 = !{null}
!64 = !DILocation(line: 1, scope: !61)
