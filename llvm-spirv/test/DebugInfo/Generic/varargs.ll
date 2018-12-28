; REQUIRES: object-emission
;
; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv -spirv-mem2reg=false
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o %t.ll

; RUN: llc -mtriple=%triple -O0 -filetype=obj %t.ll -o - | llvm-dwarfdump -v -debug-info - | FileCheck %s

; Test debug info for variadic function arguments.
; Created from tools/clang/tests/CodeGenCXX/debug-info-varargs.cpp with the
; function pointer removed.
;
; The ... parameter of variadic should be emitted as
; DW_TAG_unspecified_parameters.
;
; Normal variadic function.
; void b(int c, ...);
;
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}} "b"
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_unspecified_parameters
;
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}} "a"
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_unspecified_parameters
;
; Variadic C++ member function.
; struct A { void a(int c, ...); }
;
; ModuleID = 'debug-info-varargs.cpp'
source_filename = "debug-info-varargs.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

%struct.A = type { i8 }

; Function Attrs: noinline nounwind optnone
define spir_func void @_Z1biz(i32 %c, ...) #0 !dbg !6 {
entry:
  %c.addr = alloca i32, align 4
  %a = alloca %struct.A, align 1
  store i32 %c, i32* %c.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %c.addr, metadata !11, metadata !DIExpression()), !dbg !12
  call void @llvm.dbg.declare(metadata %struct.A* %a, metadata !13, metadata !DIExpression()), !dbg !20
  ret void, !dbg !21
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!2}
!opencl.compiler.options = !{!2}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 8.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 8.0.0 "}
!6 = distinct !DISubprogram(name: "b", linkageName: "_Z1biz", scope: !7, file: !7, line: 11, type: !8, isLocal: false, isDefinition: true, scopeLine: 11, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!7 = !DIFile(filename: "debug-info-varargs.cpp", directory: "/tmp")
!8 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !9)
!9 = !{null, !10, null}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocalVariable(name: "c", arg: 1, scope: !6, file: !7, line: 11, type: !10)
!12 = !DILocation(line: 11, scope: !6)
!13 = !DILocalVariable(name: "a", scope: !6, file: !7, line: 23, type: !14)
!14 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !7, line: 3, size: 8, flags: DIFlagTypePassByValue | DIFlagTrivial, elements: !15, identifier: "_ZTS1A")
!15 = !{!16}
!16 = !DISubprogram(name: "a", linkageName: "_ZN1A1aEiz", scope: !14, file: !7, line: 5, type: !17, isLocal: false, isDefinition: false, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: false)
!17 = !DISubroutineType(types: !18)
!18 = !{null, !19, !10, null}
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!20 = !DILocation(line: 23, scope: !6)
!21 = !DILocation(line: 29, scope: !6)
