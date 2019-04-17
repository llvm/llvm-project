; RUN: opt -argpromotion -verify -dse -S %s -o - | FileCheck %s

; Fix for PR33641. ArgumentPromotion removed the argument to bar but left the call to
; dbg.value which still used the removed argument.

%p_t = type i16*
%fun_t = type void (%p_t)*

define void @foo() {
  %tmp = alloca %fun_t
  store %fun_t @bar, %fun_t* %tmp
  ret void
}

define internal void @bar(%p_t %p)  {
  call void @llvm.dbg.value(metadata %p_t %p, metadata !4, metadata !5), !dbg !6
  ret void
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1)
!1 = !DIFile(filename: "test.c", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "bar", unit: !0)
!4 = !DILocalVariable(name: "p", scope: !3)
!5 = !DIExpression()
!6 = !DILocation(line: 1, column: 1, scope: !3)

; The %p argument should be removed, and the use of it in dbg.value should be
; changed to undef.
; CHECK:      define internal void @bar() {
; CHECK-NEXT:   call void @llvm.dbg.value(metadata i16* undef
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
