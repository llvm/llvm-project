; RUN: opt -O2 %s | llvm-dis > %t1
; RUN: llc -filetype=asm -o - %t1 | FileCheck %s
;
; A preserve_enum_value(), preserve_type_info() or preserve_field_info() call
; whose result is not used is not relocated, and it is removed without a crash.
;
; Source:
;   enum E { A = 4 };
;   struct S { int a; } __attribute__((preserve_access_index));
;   void f1(void) { (void)__builtin_preserve_enum_value(*(enum E *)A, 1); }
;   void f2(void) { (void)__builtin_preserve_type_info(*(struct S *)0, 1); }
;   void f3(struct S *p) { (void)__builtin_preserve_field_info(p->a, 0); }
; Compiler flag to generate IR:
;   clang -target bpf -S -O2 -g -emit-llvm -Xclang -disable-llvm-passes t.c

; CHECK-LABEL: f1:
; CHECK:       # %bb.0:
; CHECK-NOT:   {{r[0-9]+}} =
; CHECK:       exit
; CHECK-LABEL: f2:
; CHECK:       # %bb.0:
; CHECK-NOT:   {{r[0-9]+}} =
; CHECK:       exit
; CHECK-LABEL: f3:
; CHECK:       # %bb.0:
; CHECK-NOT:   {{r[0-9]+}} =
; CHECK:       exit
; CHECK:       .section .BTF.ext
; CHECK-NOT:   FieldReloc

source_filename = "t.c"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "bpf"

%struct.S = type { i32 }

@0 = private unnamed_addr constant [4 x i8] c"A:4\00", align 1

; Function Attrs: nounwind
define dso_local void @f1() #0 !dbg !16 {
  %1 = call i64 @llvm.bpf.preserve.enum.value(i32 0, ptr @0, i64 1), !dbg !19, !llvm.preserve.access.index !3
  ret void, !dbg !20
}

; Function Attrs: nounwind memory(none)
declare i64 @llvm.bpf.preserve.enum.value(i32, ptr, i64) #1

; Function Attrs: nounwind
define dso_local void @f2() #0 !dbg !21 {
  %1 = call i32 @llvm.bpf.preserve.type.info(i32 1, i64 1), !dbg !22, !llvm.preserve.access.index !23
  ret void, !dbg !27
}

; Function Attrs: nounwind memory(none)
declare i32 @llvm.bpf.preserve.type.info(i32, i64) #1

; Function Attrs: nounwind
define dso_local void @f3(ptr noundef %0) #0 !dbg !28 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8, !tbaa !34
    #dbg_declare(ptr %2, !33, !DIExpression(), !37)
  %3 = load ptr, ptr %2, align 8, !dbg !38, !tbaa !34
  %4 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.S) %3, i32 0, i32 0), !dbg !39, !llvm.preserve.access.index !23
  %5 = call i32 @llvm.bpf.preserve.field.info.p0(ptr %4, i64 0), !dbg !40
  ret void, !dbg !41
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.preserve.struct.access.index.p0.p0(ptr, i32 immarg, i32 immarg) #2

; Function Attrs: nounwind memory(none)
declare i32 @llvm.bpf.preserve.field.info.p0(ptr, i64 immarg) #1

attributes #0 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nounwind memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8, !9}
!llvm.errno.tbaa = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "/tmp")
!2 = !{!3}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E", file: !1, line: 1, baseType: !4, size: 32, elements: !5)
!4 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!5 = !{!6}
!6 = !DIEnumerator(name: "A", value: 4)
!7 = !{i32 7, !"Dwarf Version", i32 5}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 7, !"frame-pointer", i32 2}
!11 = !{!12, !13, i64 0}
!12 = !{!"__libc_errno", !13, i64 0}
!13 = !{!"int", !14, i64 0}
!14 = !{!"omnipotent char", !15, i64 0}
!15 = !{!"Simple C/C++ TBAA"}
!16 = distinct !DISubprogram(name: "f1", scope: !1, file: !1, line: 3, type: !17, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, keyInstructions: true)
!17 = !DISubroutineType(types: !18)
!18 = !{null}
!19 = !DILocation(line: 3, column: 23, scope: !16)
!20 = !DILocation(line: 3, column: 71, scope: !16, atomGroup: 1, atomRank: 1)
!21 = distinct !DISubprogram(name: "f2", scope: !1, file: !1, line: 4, type: !17, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, keyInstructions: true)
!22 = !DILocation(line: 4, column: 23, scope: !21)
!23 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !1, line: 2, size: 32, elements: !24)
!24 = !{!25}
!25 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !23, file: !1, line: 2, baseType: !26, size: 32)
!26 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!27 = !DILocation(line: 4, column: 72, scope: !21, atomGroup: 1, atomRank: 1)
!28 = distinct !DISubprogram(name: "f3", scope: !1, file: !1, line: 5, type: !29, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !32, keyInstructions: true)
!29 = !DISubroutineType(types: !30)
!30 = !{null, !31}
!31 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !23, size: 64)
!32 = !{!33}
!33 = !DILocalVariable(name: "p", arg: 1, scope: !28, file: !1, line: 5, type: !31)
!34 = !{!35, !35, i64 0}
!35 = !{!"p1 _ZTS1S", !36, i64 0}
!36 = !{!"any pointer", !14, i64 0}
!37 = !DILocation(line: 5, column: 19, scope: !28)
!38 = !DILocation(line: 5, column: 60, scope: !28)
!39 = !DILocation(line: 5, column: 63, scope: !28)
!40 = !DILocation(line: 5, column: 30, scope: !28)
!41 = !DILocation(line: 5, column: 70, scope: !28, atomGroup: 1, atomRank: 1)
