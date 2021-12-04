; RUN: llc -generate-arange-section -relocation-model=pic < %s | FileCheck %s

; CHECK:   .data
; CHECK-NOT: .section
; CHECK: .L_ZTId.DW.stub:

; CHECK:  .data
; CHECK-NEXT: .Lsec_end0:

source_filename = "test/DebugInfo/X86/arange-and-stub.ll"
target triple = "x86_64-linux-gnu"

@_ZTId = external constant i8*
@zed = global [1 x void ()*] [void ()* @bar], !dbg !0

define void @foo() !dbg !17 {
  ret void
}

define void @bar() personality i8* bitcast (void ()* @foo to i8*) !dbg !18 {
  invoke void @foo()
          to label %invoke.cont unwind label %lpad, !dbg !19

invoke.cont:                                      ; preds = %0
  ret void

lpad:                                             ; preds = %0
  %tmp1 = landingpad { i8*, i32 }
          filter [1 x i8*] [i8* bitcast (i8** @_ZTId to i8*)]
  ret void
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!15, !16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "zed", scope: !2, file: !6, line: 6, type: !7, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 3.7.0 (trunk 234308) (llvm/trunk 234310)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !5, imports: !4)
!3 = !DIFile(filename: "/Users/espindola/llvm/<stdin>", directory: "/Users/espindola/llvm/build")
!4 = !{}
!5 = !{!0}
!6 = !DIFile(filename: "/Users/espindola/llvm/test.cpp", directory: "/Users/espindola/llvm/build")
!7 = !DICompositeType(tag: DW_TAG_array_type, baseType: !8, size: 64, align: 64, elements: !13)
!8 = !DIDerivedType(tag: DW_TAG_typedef, name: "vifunc", file: !6, line: 5, baseType: !9)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64, align: 64)
!10 = !DISubroutineType(types: !11)
!11 = !{null, !12}
!12 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!13 = !{!14}
!14 = !DISubrange(count: 1)
!15 = !{i32 2, !"Dwarf Version", i32 4}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!17 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: !6, file: !6, line: 1, type: !10, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!18 = distinct !DISubprogram(name: "bar_d", linkageName: "bar", scope: !6, file: !6, line: 3, type: !10, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!19 = !DILocation(line: 0, scope: !18)

