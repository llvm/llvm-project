; RUN: llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_86 < %s | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_86 -verify-machineinstrs | %ptxas-verify %}

; CHECK: .global .align 1 .b8 __func___$__Z10foo_kernelv
; CHECK: .b64 __func__$_Z10foo_kernelv

@__func__._Z10foo_kernelv = private unnamed_addr constant [11 x i8] c"foo_kernel\00", align 1, !dbg !0

define void @_Z10foo_kernelv() !dbg !20 {
entry:
  call void @_Z6escapePKc(ptr noundef @__func__._Z10foo_kernelv) #2, !dbg !23
  ret void, !dbg !24
}

declare void @_Z6escapePKc(ptr)

!llvm.dbg.cu = !{!14}
!llvm.ident = !{!18}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(scope: null, file: !2, name: "__func__", linkageName: "__func__._Z10foo_kernelv", line: 6, type: !3, isLocal: true, isDefinition: true)
!2 = !DIFile(filename: "test_module.cu", directory: "/")
!3 = !DICompositeType(tag: DW_TAG_array_type, baseType: !4, size: 88, elements: !6)
!4 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !5)
!5 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!6 = !{!7}
!7 = !DISubrange(count: 11)
!14 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !2, producer: "clang version 20.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !16, nameTableKind: None)
!16 = !{!0}
!18 = !{!"clang version 20.0.0git"}
!20 = distinct !DISubprogram(name: "foo_kernel", linkageName: "_Z10foo_kernelv", scope: !2, file: !2, line: 4, type: !21, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !14)
!21 = !DISubroutineType(types: !22)
!22 = !{null}
!23 = !DILocation(line: 6, column: 5, scope: !20)
!24 = !DILocation(line: 7, column: 1, scope: !20)
