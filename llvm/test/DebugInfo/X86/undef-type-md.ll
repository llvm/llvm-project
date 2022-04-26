; RUN:  opt -S -ipsccp %S/undef-type-md.ll | FileCheck %s
; CHECK: llvm.nondebug.metadata = !{[[NONDEBUG_METADATA:![0-9]+]]}
; CHECK: [[NONDEBUG_METADATA]] =  distinct !{null} 
; CHECK: !DITemplateValueParameter({{.*}} value: %class.1 addrspace(1)* undef)

; ModuleID = '<stdin>'
source_filename = "test.cpp"

%"struct.1" = type <{ float, i32, i8, [3 x i8] }>
%"class.1" = type { %"struct.1" }

@extern_const = external addrspace(1) constant { { float, i32, i8 } }

; Function Attrs: convergent mustprogress norecurse
define linkonce_odr spir_func void @foo() align 2 !dbg !5 {
entry:
  %0 = alloca %"class.1", align 8
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.nondebug.metadata= !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cpp", directory: "/path/to")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "foo<param>", scope: !1, file: !1, line: 27, type: !6, scopeLine: 27, spFlags: DISPFlagDefinition, unit: !0, templateParams: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !{!9}
!9 = !DITemplateValueParameter(name: "S", type: !10, value: %"class.1" addrspace(1)* bitcast ({ { float, i32, i8 } } addrspace(1)* @extern_const to %"class.1" addrspace(1)*))
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{%"class.1" addrspace(1)* bitcast ({ { float, i32, i8 } } addrspace(1)* @extern_const to %"class.1" addrspace(1)*)}
