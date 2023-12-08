; RUN: opt -passes=argpromotion -S %s -o - | FileCheck %s

; Fix for PR33641. ArgumentPromotion removed the argument to bar but left the call to
; dbg.value which still used the removed argument.

; The %p argument should be removed, and the use of it in dbg.value should be
; changed to undef.

%fun_t = type ptr
define void @foo() {
  %a = alloca i16
  call void @bar(ptr %a)
  ret void
}

define internal void @bar(ptr %p) {
; CHECK-LABEL: define {{.*}}void @bar()
; CHECK-NEXT:    call void @llvm.dbg.value(metadata ptr undef, metadata !3, metadata !DIExpression()), !dbg !5
  call void @llvm.dbg.value(metadata ptr %p, metadata !3, metadata !DIExpression()), !dbg !5
  ret void
}

declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug)
!1 = !DIFile(filename: "test.c", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DILocalVariable(name: "p", scope: !4)
!4 = distinct !DISubprogram(name: "bar", scope: null, spFlags: DISPFlagDefinition, unit: !0)
!5 = !DILocation(line: 1, column: 1, scope: !4)
