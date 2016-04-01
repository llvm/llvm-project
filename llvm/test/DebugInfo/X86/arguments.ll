; REQUIRES: object-emission

; RUN: llc -mtriple=x86_64-unknown-unknown -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; IR generated from clang -g with the following source:
; struct foo {
;   foo(const foo&);
;   int i;
; };
;
; void func(foo f, foo g) {
;   f.i++;
; }

; CHECK: debug_info contents
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_linkage_name{{.*}}"_Z4func3fooS_"
; CHECK-NOT: NULL
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}"f"
; CHECK-NOT: NULL
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}"g"

%struct.foo = type { i32 }

; Function Attrs: nounwind uwtable
define void @_Z4func3fooS_(%struct.foo* %f, %struct.foo* %g) #0 !dbg !4 {
entry:
  call void @llvm.dbg.declare(metadata %struct.foo* %f, metadata !19, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.declare(metadata %struct.foo* %g, metadata !21, metadata !DIExpression()), !dbg !20
  %i = getelementptr inbounds %struct.foo, %struct.foo* %f, i32 0, i32 0, !dbg !22
  %0 = load i32, i32* %i, align 4, !dbg !22
  %inc = add nsw i32 %0, 1, !dbg !22
  store i32 %inc, i32* %i, align 4, !dbg !22
  ret void, !dbg !23
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!24}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.4 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "scratch.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "func", linkageName: "_Z4func3fooS_", line: 6, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 6, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "scratch.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch")
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8, !8}
!8 = !DICompositeType(tag: DW_TAG_structure_type, name: "foo", line: 1, size: 32, align: 32, file: !1, elements: !9)
!9 = !{!10, !12}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "i", line: 3, size: 32, align: 32, file: !1, scope: !8, baseType: !11)
!11 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!12 = !DISubprogram(name: "foo", line: 2, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 2, file: !1, scope: !8, type: !13)
!13 = !DISubroutineType(types: !14)
!14 = !{null, !15, !16}
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !8)
!16 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !17)
!17 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !8)
!19 = !DILocalVariable(name: "f", line: 6, arg: 1, scope: !4, file: !5, type: !8)
!20 = !DILocation(line: 6, scope: !4)
!21 = !DILocalVariable(name: "g", line: 6, arg: 2, scope: !4, file: !5, type: !8)
!22 = !DILocation(line: 7, scope: !4)
!23 = !DILocation(line: 8, scope: !4)
!24 = !{i32 1, !"Debug Info Version", i32 3}
