; RUN: opt -instcombine -S %s | FileCheck %s

%foo = type { i8 }

; Function Attrs: nounwind uwtable
define void @_ZN4llvm13ScaledNumbers10multiply64Emm() {
entry:
  %getU = alloca %foo, align 1
; This is supposed to make sure that the declare conversion, does not accidentally think the store OF
; %getU is a store TO %getU. There are valid reasons to have an llvm.dbg.value here, but if the pass
; is changed to emit such, a more specific check should be added to make sure that any llvm.dbg.value
; is correct.
; CHECK-NOT: @llvm.dbg.value(metadata %foo* %getU
  call void @llvm.dbg.declare(metadata %foo* %getU, metadata !3, metadata !6), !dbg !7
  store %foo* %getU, %foo** undef, align 8, !tbaa !8
  unreachable
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (https://github.com/llvm-mirror/clang 89dda3855cda574f355e6defa1d77bdae5053994) (llvm/trunk 257597)", isOptimized: true, runtimeVersion: 0, emissionKind: 1)
!1 = !DIFile(filename: "none", directory: ".")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DILocalVariable(name: "getU", scope: !4, file: !1, line: 25, type: !5)
!4 = distinct !DISubprogram(name: "multiply64", linkageName: "_ZN4llvm13ScaledNumbers10multiply64Emm", scope: null, file: !1, line: 22, isLocal: false, isDefinition: true, scopeLine: 23, flags: DIFlagPrototyped, isOptimized: true)
!5 = !DICompositeType(tag: DW_TAG_class_type, scope: !4, file: !1, line: 25, size: 8, align: 8)
!6 = !DIExpression()
!7 = !DILocation(line: 25, column: 8, scope: !4)
!8 = !{!9, !9, i64 0}
!9 = !{i64 0}
