; RUN: llc < %s -filetype=obj -o - | llvm-dwarfdump -

; To regenerate this file, use approximately the following C code:
; int* globl;
; void baz(int arg) {
;   int locl;
;   globl = &locl;
; }

; CHECK: DW_TAG_subprogram
; CHECK-NEXT:                DW_AT_low_pc
; CHECK-NEXT:                DW_AT_high_pc
;; Check that we fall back to the default frame base (the global)
; CHECK-NEXT:                DW_AT_frame_base	(DW_OP_WASM_location_int 0x3 0x0, DW_OP_stack_value)

; TODO: Find a more-reduced test case for The fix in WebAssemblyRegColoring

; ModuleID = 'debugtest-opt.c'
source_filename = "debugtest-opt.c"
target triple = "wasm32"

@globl = hidden local_unnamed_addr global i32* null, align 4, !dbg !0

; Function Attrs: nounwind writeonly
define hidden void @baz(i32 %arg) local_unnamed_addr #0 !dbg !12 {
entry:
  %locl = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 %arg, metadata !16, metadata !DIExpression()), !dbg !18
  %0 = bitcast i32* %locl to i8*, !dbg !19
  store i32* %locl, i32** @globl, align 4, !dbg !20, !tbaa !21
  ret void, !dbg !25
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind writeonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { nounwind readnone speculatable willreturn }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8, !9, !10}
!llvm.ident = !{!11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "globl", scope: !2, file: !3, line: 2, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 11.0.0 (https://github.com/llvm/llvm-project.git ee80f8bef31e0f98c9a0e1d79dc5f1ff51ed9e3a)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "debugtest-opt.c", directory: "/s/llvm-upstream")
!4 = !{}
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 32)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{i32 7, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git ee80f8bef31e0f98c9a0e1d79dc5f1ff51ed9e3a)"}
!12 = distinct !DISubprogram(name: "baz", scope: !3, file: !3, line: 14, type: !13, scopeLine: 14, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !15)
!13 = !DISubroutineType(types: !14)
!14 = !{null, !7}
!15 = !{!16, !17}
!16 = !DILocalVariable(name: "arg", arg: 1, scope: !12, file: !3, line: 14, type: !7)
!17 = !DILocalVariable(name: "locl", scope: !12, file: !3, line: 15, type: !7)
!18 = !DILocation(line: 0, scope: !12)
!19 = !DILocation(line: 15, column: 3, scope: !12)
!20 = !DILocation(line: 16, column: 9, scope: !12)
!21 = !{!22, !22, i64 0}
!22 = !{!"any pointer", !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C/C++ TBAA"}
!25 = !DILocation(line: 17, column: 1, scope: !12)
