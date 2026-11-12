; RUN: llvm-dis -o - %s.bc | FileCheck %s

; CHECK: DIModule(scope: !4, name: "dummy", file: !3, line: 2)

; ModuleID = 'DIModule-fortran-module.bc'
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
!llvm.dbg.cu = !{!4}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "foo", scope: !2, file: !3, type: !7, isLocal: false, isDefinition: true)
!2 = !DIModule(scope: !4, name: "dummy", file: !3, line: 2)
!3 = !DIFile(filename: "module.f90", directory: "/fortran")
!4 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !5)
!5 = !{}
!6 = !{!0}
!7 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
