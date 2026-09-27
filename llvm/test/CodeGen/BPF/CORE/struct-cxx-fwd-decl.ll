; RUN: not opt -O2 %s -o /dev/null 2>&1 | FileCheck %s

; IR from a producer that only declares a C++ record in the debug info has no
; DI element for a CO-RE access to one of its fields. The access is reported
; as an error instead of crashing, also when it is the parent of a nested
; access, and a field info call on it is dropped.
;
; struct Base { int pad; int b; };
; struct Inner { int a, x; };
; struct V : virtual Base { int v; Inner in; };
; unsigned long get_v(V *v) {
;   return (unsigned long)__builtin_preserve_access_index(&v->v);
; }
; unsigned info_v(V *v) { return __builtin_preserve_field_info(v->v, 0); }
; unsigned long get_in_x(V *v) {
;   return (unsigned long)__builtin_preserve_access_index(&v->in.x);
; }

; CHECK: error: test.cpp:3:1: in function get_v i64 (ptr): CO-RE access index 2 does not refer to a field of 'V' in the debug info
; CHECK: error: test.cpp:4:1: in function info_v i32 (ptr): CO-RE access index 2 does not refer to a field of 'V' in the debug info
; CHECK: error: test.cpp:5:1: in function get_in_x i64 (ptr): CO-RE access index 3 does not refer to a field of 'V' in the debug info

target triple = "bpf"

%struct.V = type <{ ptr, i32, %struct.Inner, %struct.Base, [4 x i8] }>
%struct.Base = type { i32, i32 }
%struct.Inner = type { i32, i32 }

define dso_local i64 @get_v(ptr %v) !dbg !10 {
entry:
  %0 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.V) %v, i32 1, i32 2), !dbg !14, !llvm.preserve.access.index !15
  %1 = ptrtoint ptr %0 to i64, !dbg !14
  ret i64 %1, !dbg !14
}

define dso_local i32 @info_v(ptr %v) !dbg !20 {
entry:
  %0 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.V) %v, i32 1, i32 2), !dbg !21, !llvm.preserve.access.index !15
  %1 = call i32 @llvm.bpf.preserve.field.info.p0(ptr %0, i64 0), !dbg !21
  ret i32 %1, !dbg !21
}

define dso_local i64 @get_in_x(ptr %v) !dbg !30 {
entry:
  %0 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.V) %v, i32 2, i32 3), !dbg !31, !llvm.preserve.access.index !15
  %1 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.Inner) %0, i32 1, i32 1), !dbg !31, !llvm.preserve.access.index !32
  %2 = ptrtoint ptr %1 to i64, !dbg !31
  ret i64 %2, !dbg !31
}

declare ptr @llvm.preserve.struct.access.index.p0.p0(ptr, i32 immarg, i32 immarg)
declare i32 @llvm.bpf.preserve.field.info.p0(ptr, i64 immarg)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!10 = distinct !DISubprogram(name: "get_v", scope: !1, file: !1, line: 3, type: !11, scopeLine: 3, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !DISubroutineType(types: !12)
!12 = !{!13, !16}
!13 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!14 = !DILocation(line: 3, column: 1, scope: !10)
!15 = !DICompositeType(tag: DW_TAG_structure_type, name: "V", file: !1, line: 2, flags: DIFlagFwdDecl | DIFlagNonTrivial, identifier: "_ZTS1V")
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!20 = distinct !DISubprogram(name: "info_v", scope: !1, file: !1, line: 4, type: !22, scopeLine: 4, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!21 = !DILocation(line: 4, column: 1, scope: !20)
!22 = !DISubroutineType(types: !23)
!23 = !{!24, !16}
!24 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!25 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!30 = distinct !DISubprogram(name: "get_in_x", scope: !1, file: !1, line: 5, type: !11, scopeLine: 5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!31 = !DILocation(line: 5, column: 1, scope: !30)
!32 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Inner", file: !1, line: 2, size: 64, elements: !33, identifier: "_ZTS5Inner")
!33 = !{!34, !35}
!34 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !32, file: !1, line: 2, baseType: !25, size: 32)
!35 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !32, file: !1, line: 2, baseType: !25, size: 32, offset: 32)
