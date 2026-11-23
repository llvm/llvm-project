; RUN: opt -O2 %s | llc -filetype=obj -o - | \
; RUN:   llvm-objdump --no-addresses --no-show-raw-insn --no-print-imm-hex -dr - | \
; RUN:   FileCheck %s --implicit-check-not=CO-RE

; Source:
;   #define __pai __attribute__((preserve_access_index))
;   struct S { long x; int y; } __pai;
;   typedef struct S Row[3];
;   struct T { int pad; Row f[2]; } __pai;
;   #define _(x) __builtin_preserve_access_index(x)
;   long ptr_arr(Row (*p)[2]) { return _((*p)[1][2].y); }
;   long ptr(Row (*p)[2]) { return _(p[1][1][2].y); }
;   void *sub(Row (*p)[2]) { return _(&(*p)[1]); }
;   unsigned fi(Row (*p)[2]) { return __builtin_preserve_field_info((*p)[1][2].y, 0); }
;   long mem(struct T *t) { return _(t->f[1][2].y); }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm -Xclang -disable-llvm-passes test.c

; CHECK-LABEL: <ptr_arr>:
; CHECK:       w0 = *(u32 *)(r1 + 88)
; CHECK-NEXT:  CO-RE <byte_off> [{{[0-9]+}}] struct S::[5].y (5:1)
; CHECK-LABEL: <ptr>:
; CHECK:       w0 = *(u32 *)(r1 + 184)
; CHECK-NEXT:  CO-RE <byte_off> [{{[0-9]+}}] struct S::[11].y (11:1)
; CHECK-LABEL: <sub>:
; CHECK:       r{{[0-9]+}} = 48
; CHECK-NEXT:  CO-RE <byte_off> [{{[0-9]+}}] struct S::[3] (3)
; CHECK-LABEL: <fi>:
; CHECK:       r0 = 88
; CHECK-NEXT:  CO-RE <byte_off> [{{[0-9]+}}] struct S::[5].y (5:1)
; CHECK-LABEL: <mem>:
; CHECK:       w0 = *(u32 *)(r1 + 96)
; CHECK-NEXT:  CO-RE <byte_off> [{{[0-9]+}}] struct T::f[1][2].y (0:1:1:2:1)

source_filename = "test.c"
target triple = "bpf"

%struct.S = type { i64, i32 }
%struct.T = type { i32, [2 x [3 x %struct.S]] }

; Function Attrs: nounwind
define dso_local i64 @ptr_arr(ptr noundef %0) #0 !dbg !26 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8, !tbaa !31
    #dbg_declare(ptr %2, !30, !DIExpression(), !34)
  %3 = load ptr, ptr %2, align 8, !dbg !35, !tbaa !31
  %4 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([2 x [3 x %struct.S]]) %3, i32 1, i32 1), !dbg !36, !llvm.preserve.access.index !14
  %5 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x %struct.S]) %4, i32 1, i32 2), !dbg !36, !llvm.preserve.access.index !3
  %6 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.S) %5, i32 1, i32 1), !dbg !37, !llvm.preserve.access.index !5
  %7 = load i32, ptr %6, align 8, !dbg !38, !tbaa !39
  %8 = sext i32 %7 to i64, !dbg !42
  ret i64 %8, !dbg !43
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.preserve.array.access.index.p0.p0(ptr, i32 immarg, i32 immarg) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.preserve.struct.access.index.p0.p0(ptr, i32 immarg, i32 immarg) #1

; Function Attrs: nounwind
define dso_local i64 @ptr(ptr noundef %0) #0 !dbg !44 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8, !tbaa !31
    #dbg_declare(ptr %2, !46, !DIExpression(), !47)
  %3 = load ptr, ptr %2, align 8, !dbg !48, !tbaa !31
  %4 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([2 x [3 x %struct.S]]) %3, i32 0, i32 1), !dbg !48, !llvm.preserve.access.index !13
  %5 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([2 x [3 x %struct.S]]) %4, i32 1, i32 1), !dbg !48, !llvm.preserve.access.index !14
  %6 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x %struct.S]) %5, i32 1, i32 2), !dbg !48, !llvm.preserve.access.index !3
  %7 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.S) %6, i32 1, i32 1), !dbg !49, !llvm.preserve.access.index !5
  %8 = load i32, ptr %7, align 8, !dbg !50, !tbaa !39
  %9 = sext i32 %8 to i64, !dbg !51
  ret i64 %9, !dbg !52
}

; Function Attrs: nounwind
define dso_local ptr @sub(ptr noundef %0) #0 !dbg !53 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8, !tbaa !31
    #dbg_declare(ptr %2, !58, !DIExpression(), !59)
  %3 = load ptr, ptr %2, align 8, !dbg !60, !tbaa !31
  %4 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([2 x [3 x %struct.S]]) %3, i32 1, i32 1), !dbg !61, !llvm.preserve.access.index !14
  ret ptr %4, !dbg !62
}

; Function Attrs: nounwind
define dso_local i32 @fi(ptr noundef %0) #0 !dbg !63 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8, !tbaa !31
    #dbg_declare(ptr %2, !68, !DIExpression(), !69)
  %3 = load ptr, ptr %2, align 8, !dbg !70, !tbaa !31
  %4 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([2 x [3 x %struct.S]]) %3, i32 1, i32 1), !dbg !71, !llvm.preserve.access.index !14
  %5 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x %struct.S]) %4, i32 1, i32 2), !dbg !71, !llvm.preserve.access.index !3
  %6 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.S) %5, i32 1, i32 1), !dbg !72, !llvm.preserve.access.index !5
  %7 = call i32 @llvm.bpf.preserve.field.info.p0(ptr %6, i64 0), !dbg !73
  ret i32 %7, !dbg !74
}

; Function Attrs: nounwind memory(none)
declare i32 @llvm.bpf.preserve.field.info.p0(ptr, i64 immarg) #2

; Function Attrs: nounwind
define dso_local i64 @mem(ptr noundef %0) #0 !dbg !75 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8, !tbaa !85
    #dbg_declare(ptr %2, !84, !DIExpression(), !87)
  %3 = load ptr, ptr %2, align 8, !dbg !88, !tbaa !85
  %4 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.T) %3, i32 1, i32 1), !dbg !89, !llvm.preserve.access.index !79
  %5 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([2 x [3 x %struct.S]]) %4, i32 1, i32 1), !dbg !88, !llvm.preserve.access.index !14
  %6 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x %struct.S]) %5, i32 1, i32 2), !dbg !88, !llvm.preserve.access.index !3
  %7 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.S) %6, i32 1, i32 1), !dbg !90, !llvm.preserve.access.index !5
  %8 = load i32, ptr %7, align 8, !dbg !91, !tbaa !39
  %9 = sext i32 %8 to i64, !dbg !92
  ret i64 %9, !dbg !93
}

attributes #0 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nounwind memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!17, !18, !19}
!llvm.ident = !{!20}
!llvm.errno.tbaa = !{!21}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{!3, !13, !14}
!3 = !DIDerivedType(tag: DW_TAG_typedef, name: "Row", file: !1, line: 3, baseType: !4)
!4 = !DICompositeType(tag: DW_TAG_array_type, baseType: !5, size: 384, elements: !11)
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !1, line: 2, size: 128, elements: !6)
!6 = !{!7, !9}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !5, file: !1, line: 2, baseType: !8, size: 64)
!8 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !5, file: !1, line: 2, baseType: !10, size: 32, offset: 64)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DISubrange(count: 3)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!14 = !DICompositeType(tag: DW_TAG_array_type, baseType: !3, size: 768, elements: !15)
!15 = !{!16}
!16 = !DISubrange(count: 2)
!17 = !{i32 7, !"Dwarf Version", i32 5}
!18 = !{i32 2, !"Debug Info Version", i32 3}
!19 = !{i32 7, !"frame-pointer", i32 2}
!20 = !{!"clang version 24.0.0git"}
!21 = !{!22, !23, i64 0}
!22 = !{!"__libc_errno", !23, i64 0}
!23 = !{!"int", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C/C++ TBAA"}
!26 = distinct !DISubprogram(name: "ptr_arr", scope: !1, file: !1, line: 6, type: !27, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !29, keyInstructions: true)
!27 = !DISubroutineType(types: !28)
!28 = !{!8, !13}
!29 = !{!30}
!30 = !DILocalVariable(name: "p", arg: 1, scope: !26, file: !1, line: 6, type: !13)
!31 = !{!32, !32, i64 0}
!32 = !{!"p1 _ZTS1S", !33, i64 0}
!33 = !{!"any pointer", !24, i64 0}
!34 = !DILocation(line: 6, column: 20, scope: !26)
!35 = !DILocation(line: 6, column: 40, scope: !26)
!36 = !DILocation(line: 6, column: 38, scope: !26)
!37 = !DILocation(line: 6, column: 49, scope: !26)
!38 = !DILocation(line: 6, column: 49, scope: !26, atomGroup: 1, atomRank: 3)
!39 = !{!40, !23, i64 8}
!40 = !{!"S", !41, i64 0, !23, i64 8}
!41 = !{!"long", !24, i64 0}
!42 = !DILocation(line: 6, column: 36, scope: !26, atomGroup: 1, atomRank: 2)
!43 = !DILocation(line: 6, column: 29, scope: !26, atomGroup: 1, atomRank: 1)
!44 = distinct !DISubprogram(name: "ptr", scope: !1, file: !1, line: 7, type: !27, scopeLine: 7, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !45, keyInstructions: true)
!45 = !{!46}
!46 = !DILocalVariable(name: "p", arg: 1, scope: !44, file: !1, line: 7, type: !13)
!47 = !DILocation(line: 7, column: 16, scope: !44)
!48 = !DILocation(line: 7, column: 34, scope: !44)
!49 = !DILocation(line: 7, column: 45, scope: !44)
!50 = !DILocation(line: 7, column: 45, scope: !44, atomGroup: 1, atomRank: 3)
!51 = !DILocation(line: 7, column: 32, scope: !44, atomGroup: 1, atomRank: 2)
!52 = !DILocation(line: 7, column: 25, scope: !44, atomGroup: 1, atomRank: 1)
!53 = distinct !DISubprogram(name: "sub", scope: !1, file: !1, line: 8, type: !54, scopeLine: 8, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !57, keyInstructions: true)
!54 = !DISubroutineType(types: !55)
!55 = !{!56, !13}
!56 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!57 = !{!58}
!58 = !DILocalVariable(name: "p", arg: 1, scope: !53, file: !1, line: 8, type: !13)
!59 = !DILocation(line: 8, column: 17, scope: !53)
!60 = !DILocation(line: 8, column: 38, scope: !53)
!61 = !DILocation(line: 8, column: 36, scope: !53, atomGroup: 1, atomRank: 2)
!62 = !DILocation(line: 8, column: 26, scope: !53, atomGroup: 1, atomRank: 1)
!63 = distinct !DISubprogram(name: "fi", scope: !1, file: !1, line: 9, type: !64, scopeLine: 9, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !67, keyInstructions: true)
!64 = !DISubroutineType(types: !65)
!65 = !{!66, !13}
!66 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!67 = !{!68}
!68 = !DILocalVariable(name: "p", arg: 1, scope: !63, file: !1, line: 9, type: !13)
!69 = !DILocation(line: 9, column: 19, scope: !63)
!70 = !DILocation(line: 9, column: 67, scope: !63)
!71 = !DILocation(line: 9, column: 65, scope: !63)
!72 = !DILocation(line: 9, column: 76, scope: !63)
!73 = !DILocation(line: 9, column: 35, scope: !63, atomGroup: 1, atomRank: 2)
!74 = !DILocation(line: 9, column: 28, scope: !63, atomGroup: 1, atomRank: 1)
!75 = distinct !DISubprogram(name: "mem", scope: !1, file: !1, line: 10, type: !76, scopeLine: 10, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !83, keyInstructions: true)
!76 = !DISubroutineType(types: !77)
!77 = !{!8, !78}
!78 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !79, size: 64)
!79 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "T", file: !1, line: 4, size: 832, elements: !80)
!80 = !{!81, !82}
!81 = !DIDerivedType(tag: DW_TAG_member, name: "pad", scope: !79, file: !1, line: 4, baseType: !10, size: 32)
!82 = !DIDerivedType(tag: DW_TAG_member, name: "f", scope: !79, file: !1, line: 4, baseType: !14, size: 768, offset: 64)
!83 = !{!84}
!84 = !DILocalVariable(name: "t", arg: 1, scope: !75, file: !1, line: 10, type: !78)
!85 = !{!86, !86, i64 0}
!86 = !{!"p1 _ZTS1T", !33, i64 0}
!87 = !DILocation(line: 10, column: 20, scope: !75)
!88 = !DILocation(line: 10, column: 34, scope: !75)
!89 = !DILocation(line: 10, column: 37, scope: !75)
!90 = !DILocation(line: 10, column: 45, scope: !75)
!91 = !DILocation(line: 10, column: 45, scope: !75, atomGroup: 1, atomRank: 3)
!92 = !DILocation(line: 10, column: 32, scope: !75, atomGroup: 1, atomRank: 2)
!93 = !DILocation(line: 10, column: 25, scope: !75, atomGroup: 1, atomRank: 1)
