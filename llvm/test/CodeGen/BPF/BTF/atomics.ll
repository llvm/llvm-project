; RUN: llc -mtriple=bpfel -mcpu=v3 -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck -check-prefixes=CHECK %s
;
; Source:
;   #include <stdatomic.h>
;   struct gstruct_t {
;     _Atomic int a;
;   } gstruct;
;   extern _Atomic int ext;
;   _Atomic int gbl;
;   _Atomic int *pgbl;
;   volatile _Atomic int vvar;
;   _Atomic int __attribute__((btf_type_tag("foo"))) *tagptr1;
;   volatile __attribute__((btf_type_tag("foo"))) _Atomic int *tagptr2;
;   _Atomic int foo(_Atomic int a1, _Atomic int *p1) {
;     (void)__c11_atomic_fetch_add(&gstruct.a, 1, memory_order_relaxed);
;     (void)__c11_atomic_fetch_add(&ext, 1, memory_order_relaxed);
;     (void)__c11_atomic_fetch_add(&gbl, 1, memory_order_relaxed);
;     (void)__c11_atomic_fetch_add(pgbl, 1, memory_order_relaxed);
;     (void)__c11_atomic_fetch_add(&vvar, 1, memory_order_relaxed);
;     (void)__c11_atomic_fetch_add(p1, 1, memory_order_relaxed);
;
;     return a1;
;   }

target triple = "bpf"

%struct.gstruct_t = type { i32 }

@gstruct = dso_local global %struct.gstruct_t zeroinitializer, align 4, !dbg !0
@ext = external dso_local global i32, align 4, !dbg !34
@gbl = dso_local global i32 0, align 4, !dbg !16
@pgbl = dso_local local_unnamed_addr global ptr null, align 8, !dbg !20
@vvar = dso_local global i32 0, align 4, !dbg !23
@tagptr1 = dso_local local_unnamed_addr global ptr null, align 8, !dbg !26
@tagptr2 = dso_local local_unnamed_addr global ptr null, align 8, !dbg !31

; Function Attrs: mustprogress nofree norecurse nounwind willreturn
define dso_local i32 @foo(i32 returned %a1, ptr nocapture noundef %p1) local_unnamed_addr #0 !dbg !45 {
entry:
    #dbg_value(i32 %a1, !49, !DIExpression(), !51)
    #dbg_value(ptr %p1, !50, !DIExpression(), !51)
  %0 = atomicrmw add ptr @gstruct, i32 1 monotonic, align 4, !dbg !52
  %1 = atomicrmw add ptr @ext, i32 1 monotonic, align 4, !dbg !53
  %2 = atomicrmw add ptr @gbl, i32 1 monotonic, align 4, !dbg !54
  %3 = load ptr, ptr @pgbl, align 8, !dbg !55, !tbaa !56
  %4 = atomicrmw add ptr %3, i32 1 monotonic, align 4, !dbg !60
  %5 = atomicrmw volatile add ptr @vvar, i32 1 monotonic, align 4, !dbg !61
  %6 = atomicrmw add ptr %p1, i32 1 monotonic, align 4, !dbg !62
  ret i32 %a1, !dbg !63
}

; CHECK:             [1] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK-NEXT:        [2] PTR '(anon)' type_id=1
; CHECK-NEXT:        [3] FUNC_PROTO '(anon)' ret_type_id=1 vlen=2
; CHECK-NEXT:         'a1' type_id=1
; CHECK-NEXT:         'p1' type_id=2
; CHECK-NEXT:        [4] FUNC 'foo' type_id=3 linkage=global
; CHECK-NEXT:        [5] STRUCT 'gstruct_t' size=4 vlen=1
; CHECK-NEXT:         'a' type_id=1 bits_offset=0
; CHECK-NEXT:        [6] VAR 'gstruct' type_id=5, linkage=global
; CHECK-NEXT:        [7] VAR 'ext' type_id=1, linkage=extern
; CHECK-NEXT:        [8] VAR 'gbl' type_id=1, linkage=global
; CHECK-NEXT:        [9] VAR 'pgbl' type_id=2, linkage=global
; CHECK-NEXT:        [10] VOLATILE '(anon)' type_id=1
; CHECK-NEXT:        [11] VAR 'vvar' type_id=10, linkage=global
; CHECK-NEXT:        [12] TYPE_TAG 'foo' type_id=1
; CHECK-NEXT:        [13] PTR '(anon)' type_id=12
; CHECK-NEXT:        [14] VAR 'tagptr1' type_id=13, linkage=global
; CHECK-NEXT:        [15] TYPE_TAG 'foo' type_id=10
; CHECK-NEXT:        [16] PTR '(anon)' type_id=15
; CHECK-NEXT:        [17] VAR 'tagptr2' type_id=16, linkage=global
; CHECK-NEXT:        [18] DATASEC '.bss' size=0 vlen=6
; CHECK-NEXT:         type_id=6 offset=0 size=4
; CHECK-NEXT:         type_id=8 offset=0 size=4
; CHECK-NEXT:         type_id=9 offset=0 size=8
; CHECK-NEXT:         type_id=11 offset=0 size=4
; CHECK-NEXT:         type_id=14 offset=0 size=8
; CHECK-NEXT:         type_id=17 offset=0 size=8

attributes #0 = { mustprogress nofree norecurse nounwind willreturn "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!39, !40, !41, !42, !43}
!llvm.ident = !{!44}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "gstruct", scope: !2, file: !3, line: 4, type: !36, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 20.0.0git (git@github.com:yonghong-song/llvm-project.git 96b5b6e527c024bea84f07ea11d4b3ff63468c22)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !15, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test6.c", directory: "/tmp/home/yhs/tmp3", checksumkind: CSK_MD5, checksum: "e743f2985da6027dcc5e048bd1dcccca")
!4 = !{!5}
!5 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "memory_order", file: !6, line: 68, baseType: !7, size: 32, elements: !8)
!6 = !DIFile(filename: "work/yhs/llvm-project/llvm/build/install/lib/clang/20/include/stdatomic.h", directory: "/home/yhs", checksumkind: CSK_MD5, checksum: "f17199a988fe91afffaf0f943ef87096")
!7 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!8 = !{!9, !10, !11, !12, !13, !14}
!9 = !DIEnumerator(name: "memory_order_relaxed", value: 0)
!10 = !DIEnumerator(name: "memory_order_consume", value: 1)
!11 = !DIEnumerator(name: "memory_order_acquire", value: 2)
!12 = !DIEnumerator(name: "memory_order_release", value: 3)
!13 = !DIEnumerator(name: "memory_order_acq_rel", value: 4)
!14 = !DIEnumerator(name: "memory_order_seq_cst", value: 5)
!15 = !{!0, !16, !20, !23, !26, !31, !34}
!16 = !DIGlobalVariableExpression(var: !17, expr: !DIExpression())
!17 = distinct !DIGlobalVariable(name: "gbl", scope: !2, file: !3, line: 6, type: !18, isLocal: false, isDefinition: true)
!18 = !DIDerivedType(tag: DW_TAG_atomic_type, baseType: !19)
!19 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!20 = !DIGlobalVariableExpression(var: !21, expr: !DIExpression())
!21 = distinct !DIGlobalVariable(name: "pgbl", scope: !2, file: !3, line: 7, type: !22, isLocal: false, isDefinition: true)
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 64)
!23 = !DIGlobalVariableExpression(var: !24, expr: !DIExpression())
!24 = distinct !DIGlobalVariable(name: "vvar", scope: !2, file: !3, line: 8, type: !25, isLocal: false, isDefinition: true)
!25 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !18)
!26 = !DIGlobalVariableExpression(var: !27, expr: !DIExpression())
!27 = distinct !DIGlobalVariable(name: "tagptr1", scope: !2, file: !3, line: 9, type: !28, isLocal: false, isDefinition: true)
!28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 64, annotations: !29)
!29 = !{!30}
!30 = !{!"btf_type_tag", !"foo"}
!31 = !DIGlobalVariableExpression(var: !32, expr: !DIExpression())
!32 = distinct !DIGlobalVariable(name: "tagptr2", scope: !2, file: !3, line: 10, type: !33, isLocal: false, isDefinition: true)
!33 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !25, size: 64, annotations: !29)
!34 = !DIGlobalVariableExpression(var: !35, expr: !DIExpression())
!35 = distinct !DIGlobalVariable(name: "ext", scope: !2, file: !3, line: 5, type: !18, isLocal: false, isDefinition: false)
!36 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "gstruct_t", file: !3, line: 2, size: 32, elements: !37)
!37 = !{!38}
!38 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !36, file: !3, line: 3, baseType: !18, size: 32)
!39 = !{i32 7, !"Dwarf Version", i32 5}
!40 = !{i32 2, !"Debug Info Version", i32 3}
!41 = !{i32 1, !"wchar_size", i32 4}
!42 = !{i32 7, !"frame-pointer", i32 2}
!43 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!44 = !{!"clang version 20.0.0git (git@github.com:yonghong-song/llvm-project.git 96b5b6e527c024bea84f07ea11d4b3ff63468c22)"}
!45 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 11, type: !46, scopeLine: 11, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !48)
!46 = !DISubroutineType(types: !47)
!47 = !{!18, !18, !22}
!48 = !{!49, !50}
!49 = !DILocalVariable(name: "a1", arg: 1, scope: !45, file: !3, line: 11, type: !18)
!50 = !DILocalVariable(name: "p1", arg: 2, scope: !45, file: !3, line: 11, type: !22)
!51 = !DILocation(line: 0, scope: !45)
!52 = !DILocation(line: 12, column: 9, scope: !45)
!53 = !DILocation(line: 13, column: 9, scope: !45)
!54 = !DILocation(line: 14, column: 9, scope: !45)
!55 = !DILocation(line: 15, column: 32, scope: !45)
!56 = !{!57, !57, i64 0}
!57 = !{!"any pointer", !58, i64 0}
!58 = !{!"omnipotent char", !59, i64 0}
!59 = !{!"Simple C/C++ TBAA"}
!60 = !DILocation(line: 15, column: 9, scope: !45)
!61 = !DILocation(line: 16, column: 9, scope: !45)
!62 = !DILocation(line: 17, column: 9, scope: !45)
!63 = !DILocation(line: 19, column: 3, scope: !45)
