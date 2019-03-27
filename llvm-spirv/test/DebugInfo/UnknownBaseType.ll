; REQUIRES: object-emission

; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv -spirv-mem2reg=false
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o %t.ll

; RUN: llc -mtriple=%triple < %t.ll -filetype=obj | llvm-dwarfdump -debug-info - | FileCheck %s

; Check that translator doesn't crash when it encounter basic type encoding
; (e.g. DW_ATE_complex_float) which is missing in the spec.

; CHECK: DW_TAG_unspecified_type
; CHECK-NEXT: DW_AT_name ("complex")


target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Function Attrs: convergent nounwind
define spir_func void @foo({ float, float }* byval align 4 %f) #0 !dbg !7 {
entry:
  call void @llvm.dbg.declare(metadata { float, float }* %f, metadata !13, metadata !DIExpression()), !dbg !14
  ret void, !dbg !14
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!opencl.ocl.version = !{!5}
!opencl.spir.version = !{!5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0 (https://llvm.org/git/clang 1b09e8845172eccc47c896f546fa30805da53d51) (https://llvm.org/git/llvm 384f64397f6ad95a361b72d62c07d7bac9f24163)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 2, i32 0}
!6 = !{!"clang version 9.0.0"}
!7 = distinct !DISubprogram(name: "foo", scope: !8, file: !8, line: 1, type: !9, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!8 = !DIFile(filename: "tmp.cl", directory: "/tmp")
!9 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !10)
!10 = !{null, !11}
!11 = !DIBasicType(name: "complex", size: 64, encoding: DW_ATE_complex_float)
!12 = !{!13}
!13 = !DILocalVariable(name: "f", arg: 1, scope: !7, file: !8, line: 1, type: !11)
!14 = !DILocation(line: 1, scope: !7)
