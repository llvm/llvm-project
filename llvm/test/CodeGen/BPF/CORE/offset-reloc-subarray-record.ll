; RUN: opt -O2 %s | llc -filetype=obj -o - | \
; RUN:   llvm-objdump --no-addresses --no-show-raw-insn --no-print-imm-hex -dr - | \
; RUN:   FileCheck %s --implicit-check-not=CO-RE

; A record inside a multi-dimensional array member stays in the access chain,
; so each access needs one relocation rooted at the outer record. This holds
; through qualifiers, typedefs, array element typedefs and nested arrays.
;
; Source:
;   #define __pai __attribute__((preserve_access_index))
;   struct S { long x; int y; } __pai;
;   typedef const volatile struct S cvS;
;   typedef struct { long x; int y; } __pai Anon;
;   typedef struct S Arr2[2][3];
;   typedef struct S Row[3];
;   union U { int i; const struct S s; } __pai;
;   union V { long l; const struct S m[2][3]; } __pai;
;   typedef union V Vrow[2];
;   struct T { int pad; struct S a[2][3]; volatile union U b[2][3]; cvS c[2][3]; Anon d[2][3]; const Arr2 e; Row f[2]; const Vrow g[3]; struct S h[2][3][4]; } __pai;
;   typedef const struct T cT;
;   int rec(cT *t) { return t->a[1][2].y; }
;   int un(cT *t) { return t->b[1][2].s.y; }
;   int cv(cT *t) { return t->c[1][2].y; }
;   int anon(cT *t) { return t->d[1][2].y; }
;   int arr2(cT *t) { return t->e[1][2].y; }
;   int row(cT *t) { return t->f[1][2].y; }
;   int nest(cT *t) { return t->g[2][1].m[1][2].y; }
;   unsigned nest_sz(cT *t) { return __builtin_preserve_field_info(t->g[2][1].m[1], 1); }
;   int var(cT *t, int n) { return t->a[n][2].y; }
;   int var3(cT *t, int n) { return t->h[n][2][3].y; }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm -Xclang -disable-llvm-passes test.c

; CHECK-LABEL: <rec>:
; CHECK:       w0 = *(u32 *)(r1 + 96)
; CHECK-NEXT:  CO-RE <byte_off> [{{[0-9]+}}] typedef cT::a[1][2].y (0:1:1:2:1)
; CHECK-LABEL: <un>:
; CHECK:       w0 = *(u32 *)(r1 + 192)
; CHECK-NEXT:  CO-RE <byte_off> [{{[0-9]+}}] typedef cT::b[1][2].s.y (0:2:1:2:1:1)
; CHECK-LABEL: <cv>:
; CHECK:       w0 = *(u32 *)(r1 + 288)
; CHECK-NEXT:  CO-RE <byte_off> [{{[0-9]+}}] typedef cT::c[1][2].y (0:3:1:2:1)
; CHECK-LABEL: <anon>:
; CHECK:       w0 = *(u32 *)(r1 + 384)
; CHECK-NEXT:  CO-RE <byte_off> [{{[0-9]+}}] typedef cT::d[1][2].y (0:4:1:2:1)
; CHECK-LABEL: <arr2>:
; CHECK:       w0 = *(u32 *)(r1 + 480)
; CHECK-NEXT:  CO-RE <byte_off> [{{[0-9]+}}] typedef cT::e[1][2].y (0:5:1:2:1)
; CHECK-LABEL: <row>:
; CHECK:       w0 = *(u32 *)(r1 + 576)
; CHECK-NEXT:  CO-RE <byte_off> [{{[0-9]+}}] typedef cT::f[1][2].y (0:6:1:2:1)
; CHECK-LABEL: <nest>:
; CHECK:       w0 = *(u32 *)(r1 + 1152)
; CHECK-NEXT:  CO-RE <byte_off> [{{[0-9]+}}] typedef cT::g[2][1].m[1][2].y (0:7:2:1:1:1:2:1)
; CHECK-LABEL: <nest_sz>:
; CHECK:       r0 = 48
; CHECK-NEXT:  CO-RE <byte_sz> [{{[0-9]+}}] typedef cT::g[2][1].m[1] (0:7:2:1:1:1)
; The variable subscript [n] is not relocated, but the constant [2] after it
; is part of the relocation rooted at struct S.
; CHECK-LABEL: <var>:
; CHECK:       r{{[0-9]+}} = 8
; CHECK-NEXT:  CO-RE <byte_off> [{{[0-9]+}}] typedef cT::a (0:1)
; CHECK:       w0 = *(u32 *)(r1 + 40)
; CHECK-NEXT:  CO-RE <byte_off> [{{[0-9]+}}] struct S::[2].y (2:1)
; In a 3-D array, both constant subscripts after [n] are folded into one
; index, 2 * 4 + 3 = 11.
; CHECK-LABEL: <var3>:
; CHECK:       r{{[0-9]+}} = 1160
; CHECK-NEXT:  CO-RE <byte_off> [{{[0-9]+}}] typedef cT::h (0:8)
; CHECK:       w0 = *(u32 *)(r1 + 184)
; CHECK-NEXT:  CO-RE <byte_off> [{{[0-9]+}}] struct S::[11].y (11:1)

source_filename = "test.c"
target triple = "bpf"

%struct.T = type { i32, [2 x [3 x %struct.S]], [2 x [3 x %union.U]], [2 x [3 x %struct.S]], [2 x [3 x %struct.Anon]], [2 x [3 x %struct.S]], [2 x [3 x %struct.S]], [3 x [2 x %union.V]], [2 x [3 x [4 x %struct.S]]] }
%union.U = type { %struct.S }
%struct.S = type { i64, i32 }
%struct.Anon = type { i64, i32 }
%union.V = type { [2 x [3 x %struct.S]] }

; Function Attrs: nounwind
define dso_local i32 @rec(ptr noundef %0) #0 !dbg !93 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8, !tbaa !99
    #dbg_declare(ptr %2, !98, !DIExpression(), !102)
  %3 = load ptr, ptr %2, align 8, !dbg !103, !tbaa !99
  %4 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.T) %3, i32 1, i32 1), !dbg !104, !llvm.preserve.access.index !3
  %5 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([2 x [3 x %struct.S]]) %4, i32 1, i32 1), !dbg !103, !llvm.preserve.access.index !62
  %6 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x %struct.S]) %5, i32 1, i32 2), !dbg !103, !llvm.preserve.access.index !63
  %7 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.S) %6, i32 1, i32 1), !dbg !105, !llvm.preserve.access.index !26
  %8 = load i32, ptr %7, align 8, !dbg !106, !tbaa !107
  ret i32 %8, !dbg !110
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.preserve.struct.access.index.p0.p0(ptr, i32 immarg, i32 immarg) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.preserve.array.access.index.p0.p0(ptr, i32 immarg, i32 immarg) #1

; Function Attrs: nounwind
define dso_local i32 @un(ptr noundef %0) #0 !dbg !111 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8, !tbaa !99
    #dbg_declare(ptr %2, !113, !DIExpression(), !114)
  %3 = load ptr, ptr %2, align 8, !dbg !115, !tbaa !99
  %4 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.T) %3, i32 2, i32 2), !dbg !116, !llvm.preserve.access.index !3
  %5 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([2 x [3 x %union.U]]) %4, i32 1, i32 1), !dbg !115, !llvm.preserve.access.index !64
  %6 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x %union.U]) %5, i32 1, i32 2), !dbg !115, !llvm.preserve.access.index !66
  %7 = call ptr @llvm.preserve.union.access.index.p0.p0(ptr %6, i32 1), !dbg !117, !llvm.preserve.access.index !118
  %8 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.S) %7, i32 1, i32 1), !dbg !119, !llvm.preserve.access.index !26
  %9 = load volatile i32, ptr %8, align 8, !dbg !120, !tbaa !121
  ret i32 %9, !dbg !122
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.preserve.union.access.index.p0.p0(ptr, i32 immarg) #1

; Function Attrs: nounwind
define dso_local i32 @cv(ptr noundef %0) #0 !dbg !123 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8, !tbaa !99
    #dbg_declare(ptr %2, !125, !DIExpression(), !126)
  %3 = load ptr, ptr %2, align 8, !dbg !127, !tbaa !99
  %4 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.T) %3, i32 3, i32 3), !dbg !128, !llvm.preserve.access.index !3
  %5 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([2 x [3 x %struct.S]]) %4, i32 1, i32 1), !dbg !127, !llvm.preserve.access.index !28
  %6 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x %struct.S]) %5, i32 1, i32 2), !dbg !127, !llvm.preserve.access.index !69
  %7 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.S) %6, i32 1, i32 1), !dbg !129, !llvm.preserve.access.index !29
  %8 = load volatile i32, ptr %7, align 8, !dbg !130, !tbaa !107
  ret i32 %8, !dbg !131
}

; Function Attrs: nounwind
define dso_local i32 @anon(ptr noundef %0) #0 !dbg !132 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8, !tbaa !99
    #dbg_declare(ptr %2, !134, !DIExpression(), !135)
  %3 = load ptr, ptr %2, align 8, !dbg !136, !tbaa !99
  %4 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.T) %3, i32 4, i32 4), !dbg !137, !llvm.preserve.access.index !3
  %5 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([2 x [3 x %struct.Anon]]) %4, i32 1, i32 1), !dbg !136, !llvm.preserve.access.index !70
  %6 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x %struct.Anon]) %5, i32 1, i32 2), !dbg !136, !llvm.preserve.access.index !71
  %7 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.Anon) %6, i32 1, i32 1), !dbg !138, !llvm.preserve.access.index !73
  %8 = load i32, ptr %7, align 8, !dbg !139, !tbaa !140
  ret i32 %8, !dbg !142
}

; Function Attrs: nounwind
define dso_local i32 @arr2(ptr noundef %0) #0 !dbg !143 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8, !tbaa !99
    #dbg_declare(ptr %2, !145, !DIExpression(), !146)
  %3 = load ptr, ptr %2, align 8, !dbg !147, !tbaa !99
  %4 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.T) %3, i32 5, i32 5), !dbg !148, !llvm.preserve.access.index !3
  %5 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([2 x [3 x %struct.S]]) %4, i32 1, i32 1), !dbg !147, !llvm.preserve.access.index !40
  %6 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x %struct.S]) %5, i32 1, i32 2), !dbg !147, !llvm.preserve.access.index !63
  %7 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.S) %6, i32 1, i32 1), !dbg !149, !llvm.preserve.access.index !26
  %8 = load i32, ptr %7, align 8, !dbg !150, !tbaa !107
  ret i32 %8, !dbg !151
}

; Function Attrs: nounwind
define dso_local i32 @row(ptr noundef %0) #0 !dbg !152 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8, !tbaa !99
    #dbg_declare(ptr %2, !154, !DIExpression(), !155)
  %3 = load ptr, ptr %2, align 8, !dbg !156, !tbaa !99
  %4 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.T) %3, i32 6, i32 6), !dbg !157, !llvm.preserve.access.index !3
  %5 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([2 x [3 x %struct.S]]) %4, i32 1, i32 1), !dbg !156, !llvm.preserve.access.index !74
  %6 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x %struct.S]) %5, i32 1, i32 2), !dbg !156, !llvm.preserve.access.index !75
  %7 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.S) %6, i32 1, i32 1), !dbg !158, !llvm.preserve.access.index !26
  %8 = load i32, ptr %7, align 8, !dbg !159, !tbaa !107
  ret i32 %8, !dbg !160
}

; Function Attrs: nounwind
define dso_local i32 @nest(ptr noundef %0) #0 !dbg !161 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8, !tbaa !99
    #dbg_declare(ptr %2, !163, !DIExpression(), !164)
  %3 = load ptr, ptr %2, align 8, !dbg !165, !tbaa !99
  %4 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.T) %3, i32 7, i32 7), !dbg !166, !llvm.preserve.access.index !3
  %5 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x [2 x %union.V]]) %4, i32 1, i32 2), !dbg !165, !llvm.preserve.access.index !49
  %6 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([2 x %union.V]) %5, i32 1, i32 1), !dbg !165, !llvm.preserve.access.index !50
  %7 = call ptr @llvm.preserve.union.access.index.p0.p0(ptr %6, i32 1), !dbg !167, !llvm.preserve.access.index !168
  %8 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([2 x [3 x %struct.S]]) %7, i32 1, i32 1), !dbg !165, !llvm.preserve.access.index !57
  %9 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x %struct.S]) %8, i32 1, i32 2), !dbg !165, !llvm.preserve.access.index !76
  %10 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.S) %9, i32 1, i32 1), !dbg !169, !llvm.preserve.access.index !26
  %11 = load i32, ptr %10, align 8, !dbg !170, !tbaa !121
  ret i32 %11, !dbg !171
}

; Function Attrs: nounwind
define dso_local i32 @nest_sz(ptr noundef %0) #0 !dbg !172 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8, !tbaa !99
    #dbg_declare(ptr %2, !177, !DIExpression(), !178)
  %3 = load ptr, ptr %2, align 8, !dbg !179, !tbaa !99
  %4 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.T) %3, i32 7, i32 7), !dbg !180, !llvm.preserve.access.index !3
  %5 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x [2 x %union.V]]) %4, i32 1, i32 2), !dbg !179, !llvm.preserve.access.index !49
  %6 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([2 x %union.V]) %5, i32 1, i32 1), !dbg !179, !llvm.preserve.access.index !50
  %7 = call ptr @llvm.preserve.union.access.index.p0.p0(ptr %6, i32 1), !dbg !181, !llvm.preserve.access.index !168
  %8 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([2 x [3 x %struct.S]]) %7, i32 1, i32 1), !dbg !179, !llvm.preserve.access.index !57
  %9 = call i32 @llvm.bpf.preserve.field.info.p0(ptr %8, i64 1), !dbg !182
  ret i32 %9, !dbg !183
}

; Function Attrs: nounwind memory(none)
declare i32 @llvm.bpf.preserve.field.info.p0(ptr, i64 immarg) #2

; Function Attrs: nounwind
define dso_local i32 @var(ptr noundef %0, i32 noundef %1) #0 !dbg !184 {
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  store ptr %0, ptr %3, align 8, !tbaa !99
    #dbg_declare(ptr %3, !188, !DIExpression(), !190)
  store i32 %1, ptr %4, align 4, !tbaa !191
    #dbg_declare(ptr %4, !189, !DIExpression(), !192)
  %5 = load ptr, ptr %3, align 8, !dbg !193, !tbaa !99
  %6 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.T) %5, i32 1, i32 1), !dbg !194, !llvm.preserve.access.index !3
  %7 = load i32, ptr %4, align 4, !dbg !195, !tbaa !191
  %8 = sext i32 %7 to i64, !dbg !193
  %9 = getelementptr inbounds [2 x [3 x %struct.S]], ptr %6, i64 0, i64 %8, !dbg !193
  %10 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x %struct.S]) %9, i32 1, i32 2), !dbg !193, !llvm.preserve.access.index !63
  %11 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.S) %10, i32 1, i32 1), !dbg !196, !llvm.preserve.access.index !26
  %12 = load i32, ptr %11, align 8, !dbg !197, !tbaa !107
  ret i32 %12, !dbg !198
}

; Function Attrs: nounwind
define dso_local i32 @var3(ptr noundef %0, i32 noundef %1) #0 !dbg !199 {
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  store ptr %0, ptr %3, align 8, !tbaa !99
    #dbg_declare(ptr %3, !201, !DIExpression(), !203)
  store i32 %1, ptr %4, align 4, !tbaa !191
    #dbg_declare(ptr %4, !202, !DIExpression(), !204)
  %5 = load ptr, ptr %3, align 8, !dbg !205, !tbaa !99
  %6 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.T) %5, i32 8, i32 8), !dbg !206, !llvm.preserve.access.index !3
  %7 = load i32, ptr %4, align 4, !dbg !207, !tbaa !191
  %8 = sext i32 %7 to i64, !dbg !205
  %9 = getelementptr inbounds [2 x [3 x [4 x %struct.S]]], ptr %6, i64 0, i64 %8, !dbg !205
  %10 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x [4 x %struct.S]]) %9, i32 1, i32 2), !dbg !205, !llvm.preserve.access.index !77
  %11 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([4 x %struct.S]) %10, i32 1, i32 3), !dbg !205, !llvm.preserve.access.index !80
  %12 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.S) %11, i32 1, i32 1), !dbg !208, !llvm.preserve.access.index !26
  %13 = load i32, ptr %12, align 8, !dbg !209, !tbaa !107
  ret i32 %13, !dbg !210
}

attributes #0 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nounwind memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!83, !84, !85, !86}
!llvm.ident = !{!87}
!llvm.errno.tbaa = !{!88}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{!3, !62, !63, !64, !66, !28, !69, !29, !70, !71, !73, !40, !74, !75, !49, !50, !57, !76, !77, !80}
!3 = !DIDerivedType(tag: DW_TAG_typedef, name: "cT", file: !1, line: 11, baseType: !4)
!4 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !5)
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "T", file: !1, line: 10, size: 12352, elements: !6)
!6 = !{!7, !9, !19, !27, !32, !39, !42, !48, !58}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "pad", scope: !5, file: !1, line: 10, baseType: !8, size: 32)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !5, file: !1, line: 10, baseType: !10, size: 768, offset: 64)
!10 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, size: 768, elements: !16)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !1, line: 2, size: 128, elements: !12)
!12 = !{!13, !15}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !11, file: !1, line: 2, baseType: !14, size: 64)
!14 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !11, file: !1, line: 2, baseType: !8, size: 32, offset: 64)
!16 = !{!17, !18}
!17 = !DISubrange(count: 2)
!18 = !DISubrange(count: 3)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !5, file: !1, line: 10, baseType: !20, size: 768, offset: 832)
!20 = !DICompositeType(tag: DW_TAG_array_type, baseType: !21, size: 768, elements: !16)
!21 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !22)
!22 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "U", file: !1, line: 7, size: 128, elements: !23)
!23 = !{!24, !25}
!24 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !22, file: !1, line: 7, baseType: !8, size: 32)
!25 = !DIDerivedType(tag: DW_TAG_member, name: "s", scope: !22, file: !1, line: 7, baseType: !26, size: 128)
!26 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !11)
!27 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !5, file: !1, line: 10, baseType: !28, size: 768, offset: 1600)
!28 = !DICompositeType(tag: DW_TAG_array_type, baseType: !29, size: 768, elements: !16)
!29 = !DIDerivedType(tag: DW_TAG_typedef, name: "cvS", file: !1, line: 3, baseType: !30)
!30 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !31)
!31 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !11)
!32 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !5, file: !1, line: 10, baseType: !33, size: 768, offset: 2368)
!33 = !DICompositeType(tag: DW_TAG_array_type, baseType: !34, size: 768, elements: !16)
!34 = !DIDerivedType(tag: DW_TAG_typedef, name: "Anon", file: !1, line: 4, baseType: !35)
!35 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !1, line: 4, size: 128, elements: !36)
!36 = !{!37, !38}
!37 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !35, file: !1, line: 4, baseType: !14, size: 64)
!38 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !35, file: !1, line: 4, baseType: !8, size: 32, offset: 64)
!39 = !DIDerivedType(tag: DW_TAG_member, name: "e", scope: !5, file: !1, line: 10, baseType: !40, size: 768, offset: 3136)
!40 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !41)
!41 = !DIDerivedType(tag: DW_TAG_typedef, name: "Arr2", file: !1, line: 5, baseType: !10)
!42 = !DIDerivedType(tag: DW_TAG_member, name: "f", scope: !5, file: !1, line: 10, baseType: !43, size: 768, offset: 3904)
!43 = !DICompositeType(tag: DW_TAG_array_type, baseType: !44, size: 768, elements: !47)
!44 = !DIDerivedType(tag: DW_TAG_typedef, name: "Row", file: !1, line: 6, baseType: !45)
!45 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, size: 384, elements: !46)
!46 = !{!18}
!47 = !{!17}
!48 = !DIDerivedType(tag: DW_TAG_member, name: "g", scope: !5, file: !1, line: 10, baseType: !49, size: 4608, offset: 4672)
!49 = !DICompositeType(tag: DW_TAG_array_type, baseType: !50, size: 4608, elements: !46)
!50 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !51)
!51 = !DIDerivedType(tag: DW_TAG_typedef, name: "Vrow", file: !1, line: 9, baseType: !52)
!52 = !DICompositeType(tag: DW_TAG_array_type, baseType: !53, size: 1536, elements: !47)
!53 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "V", file: !1, line: 8, size: 768, elements: !54)
!54 = !{!55, !56}
!55 = !DIDerivedType(tag: DW_TAG_member, name: "l", scope: !53, file: !1, line: 8, baseType: !14, size: 64)
!56 = !DIDerivedType(tag: DW_TAG_member, name: "m", scope: !53, file: !1, line: 8, baseType: !57, size: 768)
!57 = !DICompositeType(tag: DW_TAG_array_type, baseType: !26, size: 768, elements: !16)
!58 = !DIDerivedType(tag: DW_TAG_member, name: "h", scope: !5, file: !1, line: 10, baseType: !59, size: 3072, offset: 9280)
!59 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, size: 3072, elements: !60)
!60 = !{!17, !18, !61}
!61 = !DISubrange(count: 4)
!62 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !10)
!63 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !45)
!64 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !65)
!65 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !20)
!66 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !67)
!67 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !68)
!68 = !DICompositeType(tag: DW_TAG_array_type, baseType: !21, size: 384, elements: !46)
!69 = !DICompositeType(tag: DW_TAG_array_type, baseType: !29, size: 384, elements: !46)
!70 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !33)
!71 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !72)
!72 = !DICompositeType(tag: DW_TAG_array_type, baseType: !34, size: 384, elements: !46)
!73 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !34)
!74 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !43)
!75 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !44)
!76 = !DICompositeType(tag: DW_TAG_array_type, baseType: !26, size: 384, elements: !46)
!77 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !78)
!78 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, size: 1536, elements: !79)
!79 = !{!18, !61}
!80 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !81)
!81 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, size: 512, elements: !82)
!82 = !{!61}
!83 = !{i32 7, !"Dwarf Version", i32 5}
!84 = !{i32 2, !"Debug Info Version", i32 3}
!85 = !{i32 2, !"debugger-tuning", !"gdb"}
!86 = !{i32 7, !"frame-pointer", i32 2}
!87 = !{!"clang version 24.0.0git"}
!88 = !{!89, !90, i64 0}
!89 = !{!"__libc_errno", !90, i64 0}
!90 = !{!"int", !91, i64 0}
!91 = !{!"omnipotent char", !92, i64 0}
!92 = !{!"Simple C/C++ TBAA"}
!93 = distinct !DISubprogram(name: "rec", scope: !1, file: !1, line: 12, type: !94, scopeLine: 12, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !97, keyInstructions: true)
!94 = !DISubroutineType(types: !95)
!95 = !{!8, !96}
!96 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64)
!97 = !{!98}
!98 = !DILocalVariable(name: "t", arg: 1, scope: !93, file: !1, line: 12, type: !96)
!99 = !{!100, !100, i64 0}
!100 = !{!"p1 _ZTS1T", !101, i64 0}
!101 = !{!"any pointer", !91, i64 0}
!102 = !DILocation(line: 12, column: 13, scope: !93)
!103 = !DILocation(line: 12, column: 25, scope: !93)
!104 = !DILocation(line: 12, column: 28, scope: !93)
!105 = !DILocation(line: 12, column: 36, scope: !93)
!106 = !DILocation(line: 12, column: 36, scope: !93, atomGroup: 1, atomRank: 2)
!107 = !{!108, !90, i64 8}
!108 = !{!"S", !109, i64 0, !90, i64 8}
!109 = !{!"long", !91, i64 0}
!110 = !DILocation(line: 12, column: 18, scope: !93, atomGroup: 1, atomRank: 1)
!111 = distinct !DISubprogram(name: "un", scope: !1, file: !1, line: 13, type: !94, scopeLine: 13, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !112, keyInstructions: true)
!112 = !{!113}
!113 = !DILocalVariable(name: "t", arg: 1, scope: !111, file: !1, line: 13, type: !96)
!114 = !DILocation(line: 13, column: 12, scope: !111)
!115 = !DILocation(line: 13, column: 24, scope: !111)
!116 = !DILocation(line: 13, column: 27, scope: !111)
!117 = !DILocation(line: 13, column: 35, scope: !111)
!118 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !21)
!119 = !DILocation(line: 13, column: 37, scope: !111)
!120 = !DILocation(line: 13, column: 37, scope: !111, atomGroup: 1, atomRank: 2)
!121 = !{!91, !91, i64 0}
!122 = !DILocation(line: 13, column: 17, scope: !111, atomGroup: 1, atomRank: 1)
!123 = distinct !DISubprogram(name: "cv", scope: !1, file: !1, line: 14, type: !94, scopeLine: 14, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !124, keyInstructions: true)
!124 = !{!125}
!125 = !DILocalVariable(name: "t", arg: 1, scope: !123, file: !1, line: 14, type: !96)
!126 = !DILocation(line: 14, column: 12, scope: !123)
!127 = !DILocation(line: 14, column: 24, scope: !123)
!128 = !DILocation(line: 14, column: 27, scope: !123)
!129 = !DILocation(line: 14, column: 35, scope: !123)
!130 = !DILocation(line: 14, column: 35, scope: !123, atomGroup: 1, atomRank: 2)
!131 = !DILocation(line: 14, column: 17, scope: !123, atomGroup: 1, atomRank: 1)
!132 = distinct !DISubprogram(name: "anon", scope: !1, file: !1, line: 15, type: !94, scopeLine: 15, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !133, keyInstructions: true)
!133 = !{!134}
!134 = !DILocalVariable(name: "t", arg: 1, scope: !132, file: !1, line: 15, type: !96)
!135 = !DILocation(line: 15, column: 14, scope: !132)
!136 = !DILocation(line: 15, column: 26, scope: !132)
!137 = !DILocation(line: 15, column: 29, scope: !132)
!138 = !DILocation(line: 15, column: 37, scope: !132)
!139 = !DILocation(line: 15, column: 37, scope: !132, atomGroup: 1, atomRank: 2)
!140 = !{!141, !90, i64 8}
!141 = !{!"", !109, i64 0, !90, i64 8}
!142 = !DILocation(line: 15, column: 19, scope: !132, atomGroup: 1, atomRank: 1)
!143 = distinct !DISubprogram(name: "arr2", scope: !1, file: !1, line: 16, type: !94, scopeLine: 16, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !144, keyInstructions: true)
!144 = !{!145}
!145 = !DILocalVariable(name: "t", arg: 1, scope: !143, file: !1, line: 16, type: !96)
!146 = !DILocation(line: 16, column: 14, scope: !143)
!147 = !DILocation(line: 16, column: 26, scope: !143)
!148 = !DILocation(line: 16, column: 29, scope: !143)
!149 = !DILocation(line: 16, column: 37, scope: !143)
!150 = !DILocation(line: 16, column: 37, scope: !143, atomGroup: 1, atomRank: 2)
!151 = !DILocation(line: 16, column: 19, scope: !143, atomGroup: 1, atomRank: 1)
!152 = distinct !DISubprogram(name: "row", scope: !1, file: !1, line: 17, type: !94, scopeLine: 17, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !153, keyInstructions: true)
!153 = !{!154}
!154 = !DILocalVariable(name: "t", arg: 1, scope: !152, file: !1, line: 17, type: !96)
!155 = !DILocation(line: 17, column: 13, scope: !152)
!156 = !DILocation(line: 17, column: 25, scope: !152)
!157 = !DILocation(line: 17, column: 28, scope: !152)
!158 = !DILocation(line: 17, column: 36, scope: !152)
!159 = !DILocation(line: 17, column: 36, scope: !152, atomGroup: 1, atomRank: 2)
!160 = !DILocation(line: 17, column: 18, scope: !152, atomGroup: 1, atomRank: 1)
!161 = distinct !DISubprogram(name: "nest", scope: !1, file: !1, line: 18, type: !94, scopeLine: 18, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !162, keyInstructions: true)
!162 = !{!163}
!163 = !DILocalVariable(name: "t", arg: 1, scope: !161, file: !1, line: 18, type: !96)
!164 = !DILocation(line: 18, column: 14, scope: !161)
!165 = !DILocation(line: 18, column: 26, scope: !161)
!166 = !DILocation(line: 18, column: 29, scope: !161)
!167 = !DILocation(line: 18, column: 37, scope: !161)
!168 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !53)
!169 = !DILocation(line: 18, column: 45, scope: !161)
!170 = !DILocation(line: 18, column: 45, scope: !161, atomGroup: 1, atomRank: 2)
!171 = !DILocation(line: 18, column: 19, scope: !161, atomGroup: 1, atomRank: 1)
!172 = distinct !DISubprogram(name: "nest_sz", scope: !1, file: !1, line: 19, type: !173, scopeLine: 19, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !176, keyInstructions: true)
!173 = !DISubroutineType(types: !174)
!174 = !{!175, !96}
!175 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!176 = !{!177}
!177 = !DILocalVariable(name: "t", arg: 1, scope: !172, file: !1, line: 19, type: !96)
!178 = !DILocation(line: 19, column: 22, scope: !172)
!179 = !DILocation(line: 19, column: 64, scope: !172)
!180 = !DILocation(line: 19, column: 67, scope: !172)
!181 = !DILocation(line: 19, column: 75, scope: !172)
!182 = !DILocation(line: 19, column: 34, scope: !172, atomGroup: 1, atomRank: 2)
!183 = !DILocation(line: 19, column: 27, scope: !172, atomGroup: 1, atomRank: 1)
!184 = distinct !DISubprogram(name: "var", scope: !1, file: !1, line: 20, type: !185, scopeLine: 20, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !187, keyInstructions: true)
!185 = !DISubroutineType(types: !186)
!186 = !{!8, !96, !8}
!187 = !{!188, !189}
!188 = !DILocalVariable(name: "t", arg: 1, scope: !184, file: !1, line: 20, type: !96)
!189 = !DILocalVariable(name: "n", arg: 2, scope: !184, file: !1, line: 20, type: !8)
!190 = !DILocation(line: 20, column: 13, scope: !184)
!191 = !{!90, !90, i64 0}
!192 = !DILocation(line: 20, column: 20, scope: !184)
!193 = !DILocation(line: 20, column: 32, scope: !184)
!194 = !DILocation(line: 20, column: 35, scope: !184)
!195 = !DILocation(line: 20, column: 37, scope: !184)
!196 = !DILocation(line: 20, column: 43, scope: !184)
!197 = !DILocation(line: 20, column: 43, scope: !184, atomGroup: 1, atomRank: 2)
!198 = !DILocation(line: 20, column: 25, scope: !184, atomGroup: 1, atomRank: 1)
!199 = distinct !DISubprogram(name: "var3", scope: !1, file: !1, line: 21, type: !185, scopeLine: 21, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !200, keyInstructions: true)
!200 = !{!201, !202}
!201 = !DILocalVariable(name: "t", arg: 1, scope: !199, file: !1, line: 21, type: !96)
!202 = !DILocalVariable(name: "n", arg: 2, scope: !199, file: !1, line: 21, type: !8)
!203 = !DILocation(line: 21, column: 14, scope: !199)
!204 = !DILocation(line: 21, column: 21, scope: !199)
!205 = !DILocation(line: 21, column: 33, scope: !199)
!206 = !DILocation(line: 21, column: 36, scope: !199)
!207 = !DILocation(line: 21, column: 38, scope: !199)
!208 = !DILocation(line: 21, column: 47, scope: !199)
!209 = !DILocation(line: 21, column: 47, scope: !199, atomGroup: 1, atomRank: 2)
!210 = !DILocation(line: 21, column: 26, scope: !199, atomGroup: 1, atomRank: 1)
