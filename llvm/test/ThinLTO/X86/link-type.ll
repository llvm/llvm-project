; Check that linking with the same type in source and target modules works without asserting.

; RUN: opt -module-summary -o %t.bc %s
; RUN: opt -module-summary -o %t-src.bc %p/Inputs/link-type-src.ll
; RUN: llvm-lto2 run -o %t.out %t.bc %t-src.bc \
; RUN:   -r %t.bc,dst_f,px -r %t.bc,src_f, -r %t-src.bc,src_f,px

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%T1 = type { double, double, i32, i32 }

define void @dst_f() {
  call void @src_f(ptr byval(%T1) null), !dbg !3
  ret void
}

declare void @src_f(ptr)

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, emissionKind: FullDebug)
!2 = !DIFile(filename: "dst.cpp", directory: "")
!3 = !DILocation(line: 1, scope: !4)
!4 = distinct !DISubprogram(name: "f", scope: !2, file: !2, type: !5, spFlags: DISPFlagDefinition, unit: !1)
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "Shared", templateParams: !8, identifier: "_ZTS6Shared")
!8 = !{!9}
!9 = !DITemplateValueParameter(value: %T1 zeroinitializer)
