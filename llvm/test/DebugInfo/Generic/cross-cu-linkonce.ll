; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

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

@x = global i32 (i32)* @_Z4funci, align 8
@y = global i32 (i32)* @_Z4funci, align 8

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr i32 @_Z4funci(i32 %i) #0 !dbg !4 {
  %1 = alloca i32, align 4
  store i32 %i, i32* %1, align 4
  call void @llvm.dbg.declare(metadata i32* %1, metadata !20, metadata !DIExpression()), !dbg !21
  %2 = load i32, i32* %1, align 4, !dbg !22
  %3 = mul nsw i32 %2, 2, !dbg !22
  ret i32 %3, !dbg !22
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { inlinehint nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0, !13}
!llvm.module.flags = !{!17, !18}
!llvm.ident = !{!19, !19}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !10, imports: !2)
!1 = !DIFile(filename: "a.cpp", directory: "/tmp/dbginfo")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "func", linkageName: "_Z4funci", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 1, file: !5, scope: !6, type: !7, variables: !2)
!5 = !DIFile(filename: "func.h", directory: "/tmp/dbginfo")
!6 = !DIFile(filename: "func.h", directory: "/tmp/dbginfo")
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !9}
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !{!11}
!11 = !DIGlobalVariable(name: "x", line: 4, isLocal: false, isDefinition: true, scope: null, file: !6, type: !12, variable: i32 (i32)** @x)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !7)
!13 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 ", isOptimized: false, emissionKind: FullDebug, file: !14, enums: !2, retainedTypes: !2, subprograms: !3, globals: !15, imports: !2)
!14 = !DIFile(filename: "b.cpp", directory: "/tmp/dbginfo")
!15 = !{!16}
!16 = !DIGlobalVariable(name: "y", line: 4, isLocal: false, isDefinition: true, scope: null, file: !6, type: !12, variable: i32 (i32)** @y)
!17 = !{i32 2, !"Dwarf Version", i32 4}
!18 = !{i32 1, !"Debug Info Version", i32 3}
!19 = !{!"clang version 3.5.0 "}
!20 = !DILocalVariable(name: "i", line: 1, arg: 1, scope: !4, file: !6, type: !9)
!21 = !DILocation(line: 1, scope: !4)
!22 = !DILocation(line: 2, scope: !4)
