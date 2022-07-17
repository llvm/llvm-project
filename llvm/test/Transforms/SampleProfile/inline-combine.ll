; RUN: opt -S %s -passes='function(instcombine),sample-profile' -sample-profile-file=%S/Inputs/inline-combine.prof | FileCheck %s
; RUN: opt -S %s -passes="function(instcombine),sample-profile" -sample-profile-file=%S/Inputs/inline-combine.prof | FileCheck %s

%"class.llvm::FoldingSetNodeID" = type { %"class.llvm::SmallVector" }
%"class.llvm::SmallVector" = type { %"class.llvm::SmallVectorImpl.base", %"struct.llvm::SmallVectorStorage" }
%"class.llvm::SmallVectorImpl.base" = type { %"class.llvm::SmallVectorTemplateBase.base" }
%"class.llvm::SmallVectorTemplateBase.base" = type { %"class.llvm::SmallVectorTemplateCommon.base" }
%"class.llvm::SmallVectorTemplateCommon.base" = type <{ %"class.llvm::SmallVectorBase", %"struct.llvm::AlignedCharArrayUnion" }>
%"class.llvm::SmallVectorBase" = type { i8*, i8*, i8* }
%"struct.llvm::AlignedCharArrayUnion" = type { %"struct.llvm::AlignedCharArray" }
%"struct.llvm::AlignedCharArray" = type { [4 x i8] }
%"struct.llvm::SmallVectorStorage" = type { [31 x %"struct.llvm::AlignedCharArrayUnion"] }
%"class.llvm::SmallVectorImpl" = type { %"class.llvm::SmallVectorTemplateBase.base", [4 x i8] }

$foo = comdat any

$bar = comdat any

define void @foo(%"class.llvm::FoldingSetNodeID"* %this) #0 align 2 !dbg !3 {
  %1 = alloca %"class.llvm::FoldingSetNodeID"*, align 8
  store %"class.llvm::FoldingSetNodeID"* %this, %"class.llvm::FoldingSetNodeID"** %1, align 8
  %2 = load %"class.llvm::FoldingSetNodeID"*, %"class.llvm::FoldingSetNodeID"** %1, align 8
  %3 = getelementptr inbounds %"class.llvm::FoldingSetNodeID", %"class.llvm::FoldingSetNodeID"* %2, i32 0, i32 0
; the call should have been inlined after sample-profile pass
; CHECK-NOT: call
  call void bitcast (void (%"class.llvm::SmallVectorImpl"*)* @bar to void (%"class.llvm::SmallVector"*)*)(%"class.llvm::SmallVector"* %3), !dbg !7
  ret void
}

define void @bar(%"class.llvm::SmallVectorImpl"* %this) #0 align 2 !dbg !8 {
  ret void
}

attributes #0 = { "comdat" "use-sample-profile" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}
!llvm.dbg.cu = !{!9}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 1, !"Debug Info Version", i32 3}
!2 = !{!"clang version 3.5 "}
!3 = distinct !DISubprogram(name: "foo", scope: !4, file: !4, line: 3, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !9, retainedNodes: !6)
!4 = !DIFile(filename: "test.cc", directory: ".")
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 4, scope: !3)
!8 = distinct !DISubprogram(name: "bar", scope: !4, file: !4, line: 7, type: !5, isLocal: false, isDefinition: true, scopeLine: 7, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !9, retainedNodes: !6)
!9 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5 ", isOptimized: false, emissionKind: FullDebug, file: !4)
