; RUN: llc -filetype=obj %s -o - | llvm-dwarfdump - | FileCheck %s
;
; Generated at -Os from:
; void *foo(void *dst);
; void start() {
;   unsigned size;
;   foo(&size);
;   if (size != 0) { // Work around a bug to preserve the dbg.value.
;   }
; }

; ModuleID = 'test1.cpp'
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios"

; Function Attrs: nounwind optsize
define void @_Z5startv() #0 !dbg !4 {
entry:
  %size = alloca i32, align 4
  %0 = bitcast i32* %size to i8*, !dbg !15
  %call = call i8* @_Z3fooPv(i8* %0) #3, !dbg !15
  call void @llvm.dbg.value(metadata i32* %size, metadata !10, metadata !16), !dbg !17

  ; The *location* of the variable should be $sp+12. This tells the debugger to
  ; look up its value in [$sp+12].
  ; CHECK: .debug_info contents:
  ; CHECK: DW_TAG_variable
  ; CHECK-NEXT: DW_AT_location
  ; CHECK-NEXT: DW_OP_breg31 WSP+12
  ; CHECK-NEXT: DW_AT_name {{.*}}"size"
  ret void, !dbg !18
}

; Function Attrs: optsize
declare i8* @_Z3fooPv(i8*) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind optsize }
attributes #1 = { optsize }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind optsize }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13}
!llvm.ident = !{!14}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.6.0 (trunk 223149) (llvm/trunk 223115)", isOptimized: true, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "<stdin>", directory: "")
!2 = !{}
!4 = distinct !DISubprogram(name: "start", linkageName: "_Z5startv", line: 2, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 3, file: !5, scope: !6, type: !7, retainedNodes: !9)
!5 = !DIFile(filename: "test1.c", directory: "")
!6 = !DIFile(filename: "test1.c", directory: "")
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !{!10}
!10 = !DILocalVariable(name: "size", line: 4, scope: !4, file: !6, type: !11)
!11 = !DIBasicType(tag: DW_TAG_base_type, name: "unsigned int", size: 32, align: 32, encoding: DW_ATE_unsigned)
!12 = !{i32 2, !"Dwarf Version", i32 2}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{!"clang version 3.6.0 (trunk 223149) (llvm/trunk 223115)"}
!15 = !DILocation(line: 5, column: 3, scope: !4)
!16 = !DIExpression(DW_OP_deref)
!17 = !DILocation(line: 4, column: 12, scope: !4)
!18 = !DILocation(line: 8, column: 1, scope: !4)
