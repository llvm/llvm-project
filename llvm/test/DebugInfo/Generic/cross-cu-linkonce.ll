; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-info - | FileCheck %s

; Built from source:
; $ clang++ a.cpp b.cpp -g -c -emit-llvm
; $ llvm-link a.bc b.bc -o ab.bc
; $ cat a.cpp
; # 1 "func.h"
; inline int func(int i) {
;   return i * 2;
; }
; int (*x)(int) = &func;
; $ cat b.cpp
; # 1 "func.h"
; inline int func(int i) {
;   return i * 2;
; }
; int (*y)(int) = &func;

; CHECK: DW_TAG_compile_unit
; CHECK:   DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "func"
; CHECK: DW_TAG_compile_unit
; CHECK-NOT: DW_TAG_subprogram

source_filename = "test/DebugInfo/Generic/cross-cu-linkonce.ll"

@x = global i32 (i32)* @_Z4funci, align 8, !dbg !0
@y = global i32 (i32)* @_Z4funci, align 8, !dbg !7

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr i32 @_Z4funci(i32 %i) #0 !dbg !19 {
  %1 = alloca i32, align 4
  store i32 %i, i32* %1, align 4
  call void @llvm.dbg.declare(metadata i32* %1, metadata !20, metadata !21), !dbg !22
  %2 = load i32, i32* %1, align 4, !dbg !23
  %3 = mul nsw i32 %2, 2, !dbg !23
  ret i32 %3, !dbg !23
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { inlinehint nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!9, !13}
!llvm.module.flags = !{!16, !17}
!llvm.ident = !{!18, !18}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "x", scope: null, file: !2, line: 4, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "func.h", directory: "/tmp/dbginfo")
!3 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 64, align: 64)
!4 = !DISubroutineType(types: !5)
!5 = !{!6, !6}
!6 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression())
!8 = !DIGlobalVariable(name: "y", scope: null, file: !2, line: 4, type: !3, isLocal: false, isDefinition: true)
!9 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !10, producer: "clang version 3.5.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !11, retainedTypes: !11, globals: !12, imports: !11)
!10 = !DIFile(filename: "a.cpp", directory: "/tmp/dbginfo")
!11 = !{}
!12 = !{!0}
!13 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !14, producer: "clang version 3.5.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !11, retainedTypes: !11, globals: !15, imports: !11)
!14 = !DIFile(filename: "b.cpp", directory: "/tmp/dbginfo")
!15 = !{!7}
!16 = !{i32 2, !"Dwarf Version", i32 4}
!17 = !{i32 1, !"Debug Info Version", i32 3}
!18 = !{!"clang version 3.5.0 "}
!19 = distinct !DISubprogram(name: "func", linkageName: "_Z4funci", scope: !2, file: !2, line: 1, type: !4, isLocal: false, isDefinition: true, scopeLine: 1, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !9, variables: !11)
!20 = !DILocalVariable(name: "i", arg: 1, scope: !19, file: !2, line: 1, type: !6)
!21 = !DIExpression()
!22 = !DILocation(line: 1, scope: !19)
!23 = !DILocation(line: 2, scope: !19)

