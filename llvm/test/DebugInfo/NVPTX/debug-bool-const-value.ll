; RUN: llc < %s -asm-verbose -mattr=+ptx76 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -asm-verbose -mattr=+ptx76 | %ptxas-verify %}

target triple = "nvptx64-nvidia-cuda"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

; CHECK: {{.*}}section {{.*}}debug_info
; CHECK: {{.*}}DW_TAG_variable
; CHECK-NEXT: {{.*}} DW_AT_address_class
; CHECK-NEXT: .b8 1{{.*}} DW_AT_const_value
; CHECK-NEXT: {{.*}} DW_AT_name

define void @test() !dbg !5
{
entry:
  %arg = alloca i1
  store i1 true, i1* %arg, !dbg !6
  call void @"llvm.dbg.value"(metadata i1 true, metadata !7, metadata !8), !dbg !6
  ret void, !dbg !6
}

declare void @"llvm.dbg.value"(metadata %".1", metadata %".2", metadata %".3")

!llvm.dbg.cu = !{ !2 }
!llvm.module.flags = !{ !9, !10 }
!nvvm.annotations = !{}

!1 = !DIFile(directory: "/source/dir", filename: "test.cu")
!2 = distinct !DICompileUnit(emissionKind: FullDebug, file: !1, isOptimized: false, language: DW_LANG_C_plus_plus, runtimeVersion: 0)
!3 = !DIBasicType(encoding: DW_ATE_boolean, name: "bool", size: 8)
!4 = !DISubroutineType(types: !{null})
!5 = distinct !DISubprogram(file: !1, isDefinition: true, isLocal: false, isOptimized: false, line: 5, linkageName: "test", name: "test", scope: !1, scopeLine: 5, type: !4, unit: !2)
!6 = !DILocation(column: 1, line: 5, scope: !5)
!7 = !DILocalVariable(arg: 0, file: !1, line: 5, name: "arg", scope: !5, type: !3)
!8 = !DIExpression()
!9 = !{ i32 2, !"Dwarf Version", i32 4 }
!10 = !{ i32 2, !"Debug Info Version", i32 3 }