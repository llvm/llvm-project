; RUN: opt -O2 %s -S | FileCheck %s

; A CO-RE access into a record declared with the class keyword
; (DW_TAG_class_type) is relocated like one into a struct. The name of each
; relocation global encodes <kind>:<patched value>$<access string>.
;
; class C { public: int x; int y; unsigned a : 3; unsigned b : 5; };
; &c->y, and the byte offset of the storage of c->b

; CHECK-DAG: @"llvm.C:0:4$0:1" =
; CHECK-DAG: @"llvm.C:0:8$0:3" =

target triple = "bpf"

%class.C = type { i32, i32, i8 }

define ptr @get_y(ptr %p) {
  %r = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%class.C) %p, i32 1, i32 1), !llvm.preserve.access.index !3
  ret ptr %r
}

define i32 @b_offset(ptr %p) {
  %f = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%class.C) %p, i32 2, i32 3), !llvm.preserve.access.index !3
  %r = call i32 @llvm.bpf.preserve.field.info.p0(ptr %f, i64 0)
  ret i32 %r
}

declare ptr @llvm.preserve.struct.access.index.p0.p0(ptr, i32 immarg, i32 immarg)
declare i32 @llvm.bpf.preserve.field.info.p0(ptr, i64 immarg)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!3 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "C", file: !1, size: 96, elements: !4)
!4 = !{!5, !6, !9, !10}
!5 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !3, file: !1, baseType: !2, size: 32, flags: DIFlagPublic)
!6 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !3, file: !1, baseType: !2, size: 32, offset: 32, flags: DIFlagPublic)
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !3, file: !1, baseType: !11, size: 3, offset: 64, flags: DIFlagPublic | DIFlagBitField, extraData: i64 64)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !3, file: !1, baseType: !11, size: 5, offset: 67, flags: DIFlagPublic | DIFlagBitField, extraData: i64 64)
!11 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
