; This test checks attributes of a Fortran module.
; RUN: %llc_dwarf %s -filetype=obj -o - | \
; RUN:   llvm-dwarfdump - | FileCheck %s

; CHECK: DW_TAG_module
; CHECK-NEXT: DW_AT_name      ("dummy")
; CHECK-NEXT: DW_AT_decl_file ("/fortran{{[/\\]}}module.f90")
; CHECK-NEXT: DW_AT_decl_line (2)

; Generated from flang compiler, Fortran source to regenerate:
; module dummy
;         integer :: foo
; end module dummy

; ModuleID = '/tmp/module-b198fa.ll'
source_filename = "/tmp/module-b198fa.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct_dummy_0_ = type <{ [4 x i8] }>

@_dummy_0_ = common global %struct_dummy_0_ zeroinitializer, align 64, !dbg !0

; Function Attrs: noinline
define float @dummy_() #0 {
.L.entry:
  ret float undef
}

attributes #0 = { noinline "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" }

!llvm.module.flags = !{!8, !9}
!llvm.dbg.cu = !{!3}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "foo", scope: !2, file: !4, type: !7, isLocal: false, isDefinition: true)
!2 = !DIModule(scope: !3, name: "dummy", file: !4, line: 2)
!3 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !4, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !5)
!4 = !DIFile(filename: "module.f90", directory: "/fortran")
!5 = !{}
!6 = !{!0}
!7 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
