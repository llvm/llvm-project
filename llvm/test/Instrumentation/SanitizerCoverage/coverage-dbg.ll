; Test that coverage instrumentation does not lose debug location.

; RUN: opt < %s -sancov -sanitizer-coverage-level=1 -S -enable-new-pm=0 | FileCheck %s
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=1 -S | FileCheck %s

; C++ source:
; 1: struct A {
; 2:  int f();
; 3:  int x;
; 4: };
; 5:
; 6: int A::f() {
; 7:    return x;
; 8: }
; clang++ ../1.cc -O3 -g -S -emit-llvm  -fno-strict-aliasing
; and add sanitize_address to @_ZN1A1fEv

; Test that __sanitizer_cov_trace_pc_guard call has !dbg pointing to the opening { of A::f().
; CHECK: call void @__sanitizer_cov_trace_pc_guard(i32*{{.*}}) #{{.*}}, !dbg [[A:!.*]]
; CHECK: [[A]] = !DILocation(line: 6, scope: !{{.*}})


target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.A = type { i32 }

; Function Attrs: nounwind readonly uwtable
define i32 @_ZN1A1fEv(%struct.A* nocapture readonly %this) #0 align 2 !dbg !13 {
entry:
  tail call void @llvm.dbg.value(metadata %struct.A* %this, i64 0, metadata !15, metadata !DIExpression()), !dbg !20
  %x = getelementptr inbounds %struct.A, %struct.A* %this, i64 0, i32 0, !dbg !21
  %0 = load i32, i32* %x, align 4, !dbg !21
  ret i32 %0, !dbg !21
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { sanitize_address nounwind readonly uwtable "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!17, !18}
!llvm.ident = !{!19}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 (210251)", isOptimized: true, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "../1.cc", directory: "/code/llvm/build0")
!2 = !{}
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "A", line: 1, size: 32, align: 32, file: !1, elements: !5, identifier: "_ZTS1A")
!5 = !{!6, !8}
!6 = !DIDerivedType(tag: DW_TAG_member, name: "x", line: 3, size: 32, align: 32, file: !1, scope: !4, baseType: !7)
!7 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !DISubprogram(name: "f", linkageName: "_ZN1A1fEv", line: 2, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 2, file: !1, scope: !4, type: !9)
!9 = !DISubroutineType(types: !10)
!10 = !{!7, !11}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !4)
!13 = distinct !DISubprogram(name: "f", linkageName: "_ZN1A1fEv", line: 6, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 6, file: !1, scope: !4, type: !9, declaration: !8, retainedNodes: !14)
!14 = !{!15}
!15 = !DILocalVariable(name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !13, type: !16)
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !4)
!17 = !{i32 2, !"Dwarf Version", i32 4}
!18 = !{i32 2, !"Debug Info Version", i32 3}
!19 = !{!"clang version 3.5.0 (210251)"}
!20 = !DILocation(line: 0, scope: !13)
!21 = !DILocation(line: 7, scope: !13)
