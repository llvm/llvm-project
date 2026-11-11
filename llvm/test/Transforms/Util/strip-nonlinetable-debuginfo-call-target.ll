; RUN: opt -S -strip-debug %s | FileCheck %s
; RUN: opt -S -passes=strip-nonlinetable-debuginfo %s | FileCheck %s

; A call target for a method of a function-local class retains the containing
; subprogram and compile unit unless the attachment itself is removed.

define void @f(ptr %callee, ptr %object) !dbg !3 {
entry:
  ; CHECK-LABEL: define void @f(
  ; CHECK: call void %callee(ptr %object)
  ; CHECK-NOT: !call_target
  call void %callee(ptr %object), !dbg !7, !call_target !8
  ret void, !dbg !7
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "call-target.cpp", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{null, !6, !6}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!7 = !DILocation(line: 3, column: 1, scope: !3)
!8 = !DISubprogram(name: "target", scope: !9, file: !1, line: 2, type: !10, scopeLine: 2, containingType: !9, flags: DIFlagPrototyped, spFlags: DISPFlagVirtual | DISPFlagLocalToUnit)
!9 = !DICompositeType(tag: DW_TAG_structure_type, name: "Local", scope: !3, file: !1, line: 2, size: 8, flags: DIFlagFwdDecl)
!10 = !DISubroutineType(types: !11)
!11 = !{null, !12}
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
