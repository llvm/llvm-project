; RUN: llc -generate-arange-section -relocation-model=pic < %s | FileCheck %s

; CHECK:   .data
; CHECK-NOT: .section
; CHECK: .L_ZTId.DW.stub:

; CHECK:  .data
; CHECK-NEXT: .Lsec_end0:

target triple = "x86_64-linux-gnu"

@_ZTId = external constant i8*
@zed = global [1 x void ()*] [void ()* @bar], !dbg !11

define void @foo() !dbg !4 {
  ret void
}

define void @bar() personality i8* bitcast (void ()* @foo to i8*) !dbg !9 {
  invoke void @foo()
          to label %invoke.cont unwind label %lpad, !dbg !19

invoke.cont:                                      ; preds = %0
  ret void

lpad:                                             ; preds = %0
  %tmp1 = landingpad { i8*, i32 }
          filter [1 x i8*] [i8* bitcast (i8** @_ZTId to i8*)]
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!17, !18}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.7.0 (trunk 234308) (llvm/trunk 234310)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !10, imports: !2)
!1 = !DIFile(filename: "/Users/espindola/llvm/<stdin>", directory: "/Users/espindola/llvm/build")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: !5, file: !5, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!5 = !DIFile(filename: "/Users/espindola/llvm/test.cpp", directory: "/Users/espindola/llvm/build")
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8}
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = distinct !DISubprogram(name: "bar_d", linkageName: "bar", scope: !5, file: !5, line: 3, type: !6, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!10 = !{!11}
!11 = !DIGlobalVariable(name: "zed", scope: !0, file: !5, line: 6, type: !12, isLocal: false, isDefinition: true)
!12 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 64, align: 64, elements: !15)
!13 = !DIDerivedType(tag: DW_TAG_typedef, name: "vifunc", file: !5, line: 5, baseType: !14)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64, align: 64)
!15 = !{!16}
!16 = !DISubrange(count: 1)
!17 = !{i32 2, !"Dwarf Version", i32 4}
!18 = !{i32 2, !"Debug Info Version", i32 3}
!19 = !DILocation(line: 0, scope: !9)
