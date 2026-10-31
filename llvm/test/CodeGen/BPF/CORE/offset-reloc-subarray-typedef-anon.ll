; RUN: opt -O2 %s | llc -filetype=obj -o - | \
; RUN:   llvm-objdump --no-addresses --no-show-raw-insn --no-print-imm-hex -dr - | \
; RUN:   FileCheck %s --implicit-check-not=CO-RE

; Source:
;   typedef struct { long x; } Anon;
;   void *anon2d(Anon (*p)[2][3]) {
;     return __builtin_preserve_access_index(&(*p)[1]);
;   }
;   long anon1d(Anon (*p)[3]) {
;     return __builtin_preserve_access_index((*p)[1].x);
;   }
;   long anonconst(const Anon (*p)[3]) {
;     return __builtin_preserve_access_index((*p)[1].x);
;   }

target triple = "bpfel"

%Anon = type { i64 }

; CHECK-LABEL: <anon2d>:
; CHECK:       = 24
; CHECK-NEXT:  CO-RE <byte_off> [{{[0-9]+}}] typedef Anon::[3] (3)
define ptr @anon2d(ptr %p) !dbg !6 {
  %row = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([2 x [3 x %Anon]]) %p, i32 1, i32 1), !llvm.preserve.access.index !10
  ret ptr %row
}

; CHECK-LABEL: <anon1d>:
; CHECK:       r0 = *(u64 *)(r1 + 8)
; CHECK-NEXT:  CO-RE <byte_off> [{{[0-9]+}}] typedef Anon::[1].x (1:0)
define i64 @anon1d(ptr %p) !dbg !7 {
  %elt = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x %Anon]) %p, i32 1, i32 1), !llvm.preserve.access.index !18
  %x = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%Anon) %elt, i32 0, i32 0), !llvm.preserve.access.index !11
  %v = load i64, ptr %x
  ret i64 %v
}

; A const element still uses the typedef, not the unnamed const, as the root.
; CHECK-LABEL: <anonconst>:
; CHECK:       r0 = *(u64 *)(r1 + 8)
; CHECK-NEXT:  CO-RE <byte_off> [{{[0-9]+}}] typedef Anon::[1].x (1:0)
define i64 @anonconst(ptr %p) !dbg !19 {
  %elt = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x %Anon]) %p, i32 1, i32 1), !llvm.preserve.access.index !22
  %x = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%Anon) %elt, i32 0, i32 0), !llvm.preserve.access.index !23
  %v = load i64, ptr %x
  ret i64 %v
}

declare ptr @llvm.preserve.array.access.index.p0.p0(ptr, i32 immarg, i32 immarg)
declare ptr @llvm.preserve.struct.access.index.p0.p0(ptr, i32 immarg, i32 immarg)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}
!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "subarray-typedef-anon.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DISubroutineType(types: !{!4, !5})
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!6 = distinct !DISubprogram(name: "anon2d", scope: !1, file: !1, type: !3, spFlags: DISPFlagDefinition, unit: !0)
!7 = distinct !DISubprogram(name: "anon1d", scope: !1, file: !1, type: !8, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DISubroutineType(types: !{!14, !17})
!9 = !DISubrange(count: 2)
!10 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, size: 384, elements: !15)
!11 = !DIDerivedType(tag: DW_TAG_typedef, name: "Anon", file: !1, baseType: !12)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !1, size: 64, elements: !{!13})
!13 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !12, baseType: !14, size: 64)
!14 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!15 = !{!9, !16}
!16 = !DISubrange(count: 3)
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 64)
!18 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, size: 192, elements: !{!16})
!19 = distinct !DISubprogram(name: "anonconst", scope: !1, file: !1, type: !20, spFlags: DISPFlagDefinition, unit: !0)
!20 = !DISubroutineType(types: !{!14, !21})
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !22, size: 64)
!22 = !DICompositeType(tag: DW_TAG_array_type, baseType: !23, size: 192, elements: !{!16})
!23 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !11)
