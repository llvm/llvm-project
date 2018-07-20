; RUN: %llc_dwarf -filetype=obj < %s | llvm-dwarfdump -debug-info - | FileCheck %s
; REQUIRES: object-emission

; typedef struct __attribute__((aligned (128))) {
;   char c;
; } S0;
;
; typedef struct {
;   __attribute__((aligned (64))) char c;
; } S1;
;
; S0 s0;
;
; void f() {
;   S1 s1;
;   __attribute__((aligned (32))) int i;
; }

; CHECK: DW_TAG_typedef
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}"S0"
; CHECK: DW_TAG_structure_type
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_alignment{{.*}}128

; CHECK: DW_TAG_variable
; CHECK: DW_AT_name{{.*}}"i"
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_alignment{{.*}}32

; CHECK: DW_TAG_typedef
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}"S1"
; CHECK: DW_TAG_structure_type
; CHECK: DW_TAG_member
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}"c"
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_alignment{{.*}}64

; ModuleID = 'test.m'
source_filename = "test.m"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S0 = type { i8, [127 x i8] }
%struct.S1 = type { i8, [63 x i8] }

@s0 = common global %struct.S0 zeroinitializer, align 128, !dbg !0

; Function Attrs: nounwind uwtable
define void @f() #0 !dbg !14 {
entry:
  %s1 = alloca %struct.S1, align 64
  %i = alloca i32, align 32
  call void @llvm.dbg.declare(metadata %struct.S1* %s1, metadata !17, metadata !22), !dbg !23
  call void @llvm.dbg.declare(metadata i32* %i, metadata !24, metadata !22), !dbg !26
  ret void, !dbg !27
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !12}
!llvm.ident = !{!13}

!0 = distinct !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "s0", scope: !2, file: !3, line: 10, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_ObjC, file: !3, producer: "clang version 4.0.0 (http://llvm.org/git/clang.git 9ce5220b821054019059c2ac4a9b132c7723832d) (http://llvm.org/git/llvm.git 9a6298be89ce0359b151c0a37af2776a12c69e85)", isOptimized: false, runtimeVersion: 1, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "test.m", directory: "/tmp")
!4 = !{}
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_typedef, name: "S0", file: !3, line: 3, baseType: !7)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !3, line: 1, size: 1024, align: 1024, elements: !8)
!8 = !{!9}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !7, file: !3, line: 2, baseType: !10, size: 8)
!10 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{!"clang version 4.0.0 (http://llvm.org/git/clang.git 9ce5220b821054019059c2ac4a9b132c7723832d) (http://llvm.org/git/llvm.git 9a6298be89ce0359b151c0a37af2776a12c69e85)"}
!14 = distinct !DISubprogram(name: "f", scope: !3, file: !3, line: 12, type: !15, isLocal: false, isDefinition: true, scopeLine: 12, isOptimized: false, unit: !2, retainedNodes: !4)
!15 = !DISubroutineType(types: !16)
!16 = !{null}
!17 = !DILocalVariable(name: "s1", scope: !14, file: !3, line: 13, type: !18)
!18 = !DIDerivedType(tag: DW_TAG_typedef, name: "S1", file: !3, line: 8, baseType: !19)
!19 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !3, line: 6, size: 512, elements: !20)
!20 = !{!21}
!21 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !19, file: !3, line: 7, baseType: !10, size: 8, align: 512)
!22 = !DIExpression()
!23 = !DILocation(line: 13, column: 6, scope: !14)
!24 = !DILocalVariable(name: "i", scope: !14, file: !3, line: 14, type: !25, align: 256)
!25 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!26 = !DILocation(line: 14, column: 37, scope: !14)
!27 = !DILocation(line: 15, column: 1, scope: !14)

