; Check that DebugLoc attached to a builtin call is preserverd after translation.
; Source:
; float f(float x) {
;   return sin(x);
; }
; Command to get this IR:
; clang -cc1 -triple spir -finclude-default-header /tmp/tmp.cl -disable-llvm-passes -emit-llvm -o - -debug-info-kind=line-tables-only | opt -mem2reg -S -o test/DebugInfo/BuiltinCallLocation.ll

; RUN: llvm-as < %s > %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; Function Attrs: convergent nounwind
define spir_func float @f(float %x) #0 !dbg !8 {
entry:
; CHECK-SPIRV: Label
; CHECK-SPIRV: ExtInst {{.*}} DebugScope
; CHECK-SPIRV: ExtInst {{.*}} sin
  %call = call spir_func float @_Z3sinf(float %x) #2, !dbg !11
; CHECK-LLVM: call spir_func float @_Z3sinf(float %x) {{.*}} !dbg ![[loc:[0-9]+]]
  ret float %call, !dbg !12
}

; Function Attrs: convergent nounwind readnone
declare spir_func float @_Z3sinf(float) #1

attributes #0 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!opencl.ocl.version = !{!5}
!opencl.spir.version = !{!6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 8.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "/tmp/<stdin>", directory: "/llvm/build")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 1, i32 0}
!6 = !{i32 1, i32 2}
!7 = !{!"clang version 8.0.0 "}
!8 = distinct !DISubprogram(name: "f", scope: !9, file: !9, line: 1, type: !10, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!9 = !DIFile(filename: "/tmp/tmp.cl", directory: "/llvm/build")
!10 = !DISubroutineType(types: !2)
!11 = !DILocation(line: 2, column: 10, scope: !8)
;CHECK-LLVM: ![[loc]] = !DILocation(line: 2, column: 10, scope: !{{.*}})
!12 = !DILocation(line: 2, column: 3, scope: !8)
