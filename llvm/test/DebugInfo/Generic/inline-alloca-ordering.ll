; RUN: opt %s --passes=inline -o - -S | FileCheck %s --implicit-check-not=dbg.value
; RUN: opt %s --passes=inline -o - -S --try-experimental-debuginfo-iterators | FileCheck %s --implicit-check-not=dbg.value

;; The inliner, specially, hoists all alloca instructions into the entry block
;; of the calling function. Ensure that it doesn't accidentally transfer the
;; dbg.value intrinsic from after the alloca to somewhere else. There should be
;; one dbg.value in the resulting block after the call to ext, and before the
;; call to init.
;;
;; This becomes significant in the context of non-instruction debug-info. When
;; splicing segments of instructions around, it's typically the range from one
;; "real" instruction to another, implicitly including all the dbg.values that
;; come before the ending instruction. The inliner is a (unique!) location in
;; LLVM that builds a range of only a single instruction kind (allocas) and thus
;; doesn't transfer the dbg.value to the entry block. This needs Special
;; Handling once we get rid of debug-intrinsics.

; CHECK: declare void @llvm.dbg.value(metadata,

; CHECK:    define i32 @bar()
; CHECK-NEXT: %1 = alloca [65 x i32], align 16
; CHECK-NEXT: call void @ext()
; CHECK-NEXT: call void @llvm.lifetime.start.p0(
; CHECK-NEXT: call void @llvm.dbg.value(metadata i32 0, metadata !10, metadata !DIExpression()), !dbg !12
; CHECK-NEXT: call void @init(ptr %1)

declare void @ext()
declare void @init(ptr)
declare void @llvm.dbg.value(metadata, metadata, metadata)

define internal i32 @foo() !dbg !4 {
  %1 = alloca [65 x i32], align 16
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !14
  call void @init(ptr %1)
  %2 = load i32, ptr %1, align 4
  ret i32 %2
}

define i32 @bar() !dbg !16 {
  call void @ext()
  %1 = call i32 @foo(), !dbg !17
  ret i32 %1
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "a.cc", directory: "/tmp")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !1, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7}
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{!"clang"}
!11 = !DILocalVariable(name: "i", arg: 1, scope: !4, file: !1, line: 3, type: !7)
!12 = !DIExpression()
!14 = !DILocation(line: 4, column: 7, scope: !15)
!15 = distinct !DILexicalBlock(scope: !4, file: !1, line: 4, column: 7)
!16 = distinct !DISubprogram(name: "bar", linkageName: "bar", scope: !1, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!17 = !DILocation(line: 4, column: 7, scope: !16)
