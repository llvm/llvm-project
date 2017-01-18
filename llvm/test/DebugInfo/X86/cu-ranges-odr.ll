; RUN: llc -split-dwarf=Enable -O0 %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t
; RUN: llvm-dwarfdump -debug-dump=all %t | FileCheck %s
; RUN: llvm-readobj --relocations %t | FileCheck --check-prefix=CHECK-RELOCS %s

; From:
; class A {
; public:
;   A(int i = 0) : a(i) {}
; private:
;   int a;
; };
;
; A a;

; With function sections enabled make sure that we have a DW_AT_ranges attribute.
; CHECK: DW_AT_ranges

; Check that we have a relocation against the .debug_ranges section.
; CHECK-RELOCS: R_X86_64_32 .debug_ranges 0x0

source_filename = "test/DebugInfo/X86/cu-ranges-odr.ll"

%class.A = type { i32 }

@a = global %class.A zeroinitializer, align 4, !dbg !0
@llvm.global_ctors = appending global [1 x { i32, void ()* }] [{ i32, void ()* } { i32 65535, void ()* @_GLOBAL__I_a }]

define internal void @__cxx_global_var_init() section ".text.startup" !dbg !18 {
entry:
  call void @_ZN1AC2Ei(%class.A* @a, i32 0), !dbg !21
  ret void, !dbg !21
}

; Function Attrs: nounwind uwtable
define linkonce_odr void @_ZN1AC2Ei(%class.A* %this, i32 %i) unnamed_addr #0 align 2 !dbg !22 {
entry:
  %this.addr = alloca %class.A*, align 8
  %i.addr = alloca i32, align 4
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !23, metadata !25), !dbg !26
  store i32 %i, i32* %i.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %i.addr, metadata !27, metadata !25), !dbg !28
  %this1 = load %class.A*, %class.A** %this.addr
  %a = getelementptr inbounds %class.A, %class.A* %this1, i32 0, i32 0, !dbg !28
  %0 = load i32, i32* %i.addr, align 4, !dbg !28
  store i32 %0, i32* %a, align 4, !dbg !28
  ret void, !dbg !28
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

define internal void @_GLOBAL__I_a() section ".text.startup" !dbg !29 {
entry:
  call void @__cxx_global_var_init(), !dbg !31
  ret void, !dbg !31
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!11}
!llvm.module.flags = !{!15, !16}
!llvm.ident = !{!17}

!0 = !DIGlobalVariableExpression(var: !1)
!1 = !DIGlobalVariable(name: "a", scope: null, file: !2, line: 8, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "baz.cpp", directory: "/usr/local/google/home/echristo/tmp")
!3 = !DICompositeType(tag: DW_TAG_class_type, name: "A", file: !2, line: 1, size: 32, align: 32, elements: !4, identifier: "_ZTS1A")
!4 = !{!5, !7}
!5 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !3, file: !2, line: 5, baseType: !6, size: 32, align: 32, flags: DIFlagPrivate)
!6 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!7 = !DISubprogram(name: "A", scope: !3, file: !2, line: 3, type: !8, isLocal: false, isDefinition: false, scopeLine: 3, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10, !6}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!11 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang version 3.5 (trunk 199923) (llvm/trunk 199940)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !12, retainedTypes: !13, globals: !14, imports: !12)
!12 = !{}
!13 = !{!3}
!14 = !{!0}
!15 = !{i32 2, !"Dwarf Version", i32 4}
!16 = !{i32 1, !"Debug Info Version", i32 3}
!17 = !{!"clang version 3.5 (trunk 199923) (llvm/trunk 199940)"}
!18 = distinct !DISubprogram(name: "__cxx_global_var_init", scope: !2, file: !2, line: 8, type: !19, isLocal: true, isDefinition: true, scopeLine: 8, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !11, variables: !12)
!19 = !DISubroutineType(types: !20)
!20 = !{null}
!21 = !DILocation(line: 8, scope: !18)
!22 = distinct !DISubprogram(name: "A", linkageName: "_ZN1AC2Ei", scope: !3, file: !2, line: 3, type: !8, isLocal: false, isDefinition: true, scopeLine: 3, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !11, declaration: !7, variables: !12)
!23 = !DILocalVariable(name: "this", arg: 1, scope: !22, type: !24, flags: DIFlagArtificial | DIFlagObjectPointer)
!24 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64, align: 64)
!25 = !DIExpression()
!26 = !DILocation(line: 0, scope: !22)
!27 = !DILocalVariable(name: "i", arg: 2, scope: !22, file: !2, line: 3, type: !6)
!28 = !DILocation(line: 3, scope: !22)
!29 = distinct !DISubprogram(linkageName: "_GLOBAL__I_a", scope: !2, file: !2, line: 3, type: !30, isLocal: true, isDefinition: true, scopeLine: 3, virtualIndex: 6, flags: DIFlagArtificial, isOptimized: false, unit: !11, variables: !12)
!30 = !DISubroutineType(types: !12)
!31 = !DILocation(line: 3, scope: !29)

