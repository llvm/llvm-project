; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

source_filename = "example.c"
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir"

%struct.A = type { i32, float }

; CHECK-SPIRV: [[ext_inst_non_semantic:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-SPIRV: [[filename_str:%[0-9]+]] = OpString "/AAAAAAAAAA/BBBBBBBB/CCCCCCCCC/example.c" 
; CHECK-SPIRV-DAG: [[type_void:%[0-9]+]] = OpTypeVoid
; CHECK-SPIRV-DAG: [[type_i32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: [[dwarf_version:%[0-9]+]] = OpConstant [[type_i32]] 5 
; CHECK-SPIRV-DAG: [[debug_info_version:%[0-9]+]] = OpConstant [[type_i32]] 21 
; CHECK-SPIRV-DAG: [[source_language:%[0-9]+]] = OpConstant [[type_i32]] 3 
; CHECK-SPIRV: [[debug_source:%[0-9]+]] = OpExtInst [[type_void]] [[ext_inst_non_semantic]] DebugSource [[filename_str]]
; CHECK-SPIRV: [[debug_compiation_unit:%[0-9]+]] = OpExtInst [[type_void]] [[ext_inst_non_semantic]] DebugCompilationUnit [[source_language]] [[dwarf_version]] [[debug_source]] [[debug_info_version]]

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_func i32 @bar(i32 noundef %n) #0 !dbg !8 {
entry:
  %n.addr = alloca i32, align 4
  store i32 %n, ptr %n.addr, align 4
    #dbg_declare(ptr %n.addr, !13, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !14)
  %0 = load i32, ptr %n.addr, align 4, !dbg !15
  %1 = load i32, ptr %n.addr, align 4, !dbg !16
  %mul = mul nsw i32 %0, %1, !dbg !17
  %2 = load i32, ptr %n.addr, align 4, !dbg !18
  %add = add nsw i32 %mul, %2, !dbg !19
  ret i32 %add, !dbg !20
}

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_func i32 @foo(i32 noundef %num, ptr addrspace(4) noundef %a) #0 !dbg !21 {
entry:
  %num.addr = alloca i32, align 4
  %a.addr = alloca ptr addrspace(4), align 4
  store i32 %num, ptr %num.addr, align 4
    #dbg_declare(ptr %num.addr, !30, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !31)
  store ptr addrspace(4) %a, ptr %a.addr, align 4
    #dbg_declare(ptr %a.addr, !32, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !33)
  %0 = load i32, ptr %num.addr, align 4, !dbg !34
  %1 = load i32, ptr %num.addr, align 4, !dbg !35
  %mul = mul nsw i32 %0, %1, !dbg !36
  %2 = load ptr addrspace(4), ptr %a.addr, align 4, !dbg !37
  %a1 = getelementptr inbounds %struct.A, ptr addrspace(4) %2, i32 0, i32 0, !dbg !38
  %3 = load i32, ptr addrspace(4) %a1, align 4, !dbg !38
  %mul2 = mul nsw i32 %mul, %3, !dbg !39
  %conv = sitofp i32 %mul2 to float, !dbg !34
  %4 = load ptr addrspace(4), ptr %a.addr, align 4, !dbg !40
  %b = getelementptr inbounds %struct.A, ptr addrspace(4) %4, i32 0, i32 1, !dbg !41
  %5 = load float, ptr addrspace(4) %b, align 4, !dbg !41
  %6 = load i32, ptr %num.addr, align 4, !dbg !42
  %call = call spir_func i32 @bar(i32 noundef %6) #2, !dbg !43
  %conv4 = sitofp i32 %call to float, !dbg !43
  %7 = call float @llvm.fmuladd.f32(float %conv, float %5, float %conv4), !dbg !44
  %conv5 = fptosi float %7 to i32, !dbg !34
  ret i32 %conv5, !dbg !45
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #1

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_func void @zar() #0 !dbg !46 {
entry:
  ret void, !dbg !49
}

attributes #0 = { convergent noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!opencl.ocl.version = !{!6}
!opencl.spir.version = !{!6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_OpenCL, file: !1, producer: "clang version 19.0.0git (fffffffffffffffffffffffffffffffffffffffff zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "example.c", directory: "/AAAAAAAAAA/BBBBBBBB/CCCCCCCCC", checksumkind: CSK_MD5, checksum: "ffffffffffffffffffffffffffffffff")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{i32 3, i32 0}
!7 = !{!"clang version 19.0.0git (fffffffffffffffffffffffffffffffffffffffff zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz)"}
!8 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 6, type: !9, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !12)
!9 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !10)
!10 = !{!11, !11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{}
!13 = !DILocalVariable(name: "n", arg: 1, scope: !8, file: !1, line: 6, type: !11)
!14 = !DILocation(line: 6, column: 13, scope: !8)
!15 = !DILocation(line: 6, column: 25, scope: !8)
!16 = !DILocation(line: 6, column: 29, scope: !8)
!17 = !DILocation(line: 6, column: 27, scope: !8)
!18 = !DILocation(line: 6, column: 33, scope: !8)
!19 = !DILocation(line: 6, column: 31, scope: !8)
!20 = !DILocation(line: 6, column: 18, scope: !8)
!21 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 8, type: !22, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !12)
!22 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !23)
!23 = !{!11, !11, !24}
!24 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !25, size: 32, dwarfAddressSpace: 4)
!25 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !1, line: 1, size: 64, elements: !26)
!26 = !{!27, !28}
!27 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !25, file: !1, line: 2, baseType: !11, size: 32)
!28 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !25, file: !1, line: 3, baseType: !29, size: 32, offset: 32)
!29 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!30 = !DILocalVariable(name: "num", arg: 1, scope: !21, file: !1, line: 8, type: !11)
!31 = !DILocation(line: 8, column: 13, scope: !21)
!32 = !DILocalVariable(name: "a", arg: 2, scope: !21, file: !1, line: 8, type: !24)
!33 = !DILocation(line: 8, column: 28, scope: !21)
!34 = !DILocation(line: 8, column: 40, scope: !21)
!35 = !DILocation(line: 8, column: 46, scope: !21)
!36 = !DILocation(line: 8, column: 44, scope: !21)
!37 = !DILocation(line: 8, column: 52, scope: !21)
!38 = !DILocation(line: 8, column: 55, scope: !21)
!39 = !DILocation(line: 8, column: 50, scope: !21)
!40 = !DILocation(line: 8, column: 59, scope: !21)
!41 = !DILocation(line: 8, column: 62, scope: !21)
!42 = !DILocation(line: 8, column: 70, scope: !21)
!43 = !DILocation(line: 8, column: 66, scope: !21)
!44 = !DILocation(line: 8, column: 64, scope: !21)
!45 = !DILocation(line: 8, column: 33, scope: !21)
!46 = distinct !DISubprogram(name: "zar", scope: !1, file: !1, line: 10, type: !47, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!47 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !48)
!48 = !{null}
!49 = !DILocation(line: 10, column: 13, scope: !46)
