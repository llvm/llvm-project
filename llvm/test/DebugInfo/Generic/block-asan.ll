; RUN: opt -S -passes=asan %s | FileCheck %s
; RUN: opt --try-experimental-debuginfo-iterators -S -passes=asan %s | FileCheck %s

; The IR of this testcase is generated from the following C code:
; void bar (int);
;
; void foo() {
;   __block int x;
;   bar(x);
; }
; by compiling it with 'clang -emit-llvm -g -S' and then by manually
; adding the sanitize_address attribute to the @foo() function (so
; that ASAN accepts to instrument the function in the above opt run).

; Check that the location of the ASAN instrumented __block variable is
; correct.
; CHECK: !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 32, DW_OP_plus_uconst, 8, DW_OP_deref, DW_OP_plus_uconst, 24)

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

%struct.__block_byref_x = type { ptr, ptr, i32, i32, i32 }

; Function Attrs: nounwind ssp uwtable
define void @foo() #0 !dbg !4 {
entry:
  %x = alloca %struct.__block_byref_x, align 8
  call void @llvm.dbg.declare(metadata ptr %x, metadata !12, metadata !22), !dbg !23
  store ptr null, ptr %x, !dbg !24
  %byref.forwarding = getelementptr inbounds %struct.__block_byref_x, ptr %x, i32 0, i32 1, !dbg !24
  store ptr %x, ptr %byref.forwarding, !dbg !24
  %byref.flags = getelementptr inbounds %struct.__block_byref_x, ptr %x, i32 0, i32 2, !dbg !24
  store i32 0, ptr %byref.flags, !dbg !24
  %byref.size = getelementptr inbounds %struct.__block_byref_x, ptr %x, i32 0, i32 3, !dbg !24
  store i32 32, ptr %byref.size, !dbg !24
  %forwarding = getelementptr inbounds %struct.__block_byref_x, ptr %x, i32 0, i32 1, !dbg !25
  %0 = load ptr, ptr %forwarding, !dbg !25
  %x1 = getelementptr inbounds %struct.__block_byref_x, ptr %0, i32 0, i32 4, !dbg !25
  %1 = load i32, ptr %x1, align 4, !dbg !25
  call void @bar(i32 %1), !dbg !25
  call void @_Block_object_dispose(ptr %x, i32 8) #3, !dbg !26
  ret void, !dbg !26
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @bar(i32) #2

declare void @_Block_object_dispose(ptr, i32)

attributes #0 = { nounwind ssp uwtable sanitize_address "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9, !10}
!llvm.ident = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.6.0 (trunk 223120) (llvm/trunk 223119)", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "block.c", directory: "/tmp")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", line: 3, isLocal: false, isDefinition: true, isOptimized: false, unit: !0, scopeLine: 3, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "block.c", directory: "/tmp")
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !{i32 2, !"Dwarf Version", i32 2}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"PIC Level", i32 2}
!11 = !{!"clang version 3.6.0 (trunk 223120) (llvm/trunk 223119)"}
!12 = !DILocalVariable(name: "x", line: 4, scope: !4, file: !5, type: !13)
!13 = !DICompositeType(tag: DW_TAG_structure_type, size: 224, file: !1, scope: !5, elements: !14)
!14 = !{!15, !17, !18, !20, !21}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "__isa", size: 64, align: 64, file: !1, scope: !5, baseType: !16)
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: null)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "__forwarding", size: 64, align: 64, offset: 64, file: !1, scope: !5, baseType: !16)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "__flags", size: 32, align: 32, offset: 128, file: !1, scope: !5, baseType: !19)
!19 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!20 = !DIDerivedType(tag: DW_TAG_member, name: "__size", size: 32, align: 32, offset: 160, file: !1, scope: !5, baseType: !19)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "x", size: 32, align: 32, offset: 192, file: !1, scope: !5, baseType: !19)
!22 = !DIExpression(DW_OP_plus_uconst, 8, DW_OP_deref, DW_OP_plus_uconst, 24)
!23 = !DILocation(line: 4, column: 15, scope: !4)
!24 = !DILocation(line: 4, column: 3, scope: !4)
!25 = !DILocation(line: 5, column: 3, scope: !4)
!26 = !DILocation(line: 6, column: 1, scope: !4)
