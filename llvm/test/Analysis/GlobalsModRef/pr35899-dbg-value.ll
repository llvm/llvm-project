; RUN: opt -S -strip-debug -globals-aa -instcombine < %s | FileCheck %s
; RUN: opt -S -globals-aa -instcombine < %s | FileCheck %s

; Having debug info around shouldn't affect what globals-aa and instcombine do.

@g = global i8 0

define void @bar(i8 %p) {
   call void @llvm.dbg.value(metadata i64 0, metadata !14, metadata !DIExpression()), !dbg !15
  ret void
}

declare void @gaz(i8 %p)

define void @foo() {
  store i8 42, i8* @g, align 1
  call void @bar(i8 1)
  %_tmp = load i8, i8* @g, align 1
  call void @gaz(i8 %_tmp)
  ret void
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!5}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = !DIFile(filename: "foo.c", directory: "/tmp")
!1 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint64_t", file: !2, line: 77, baseType: !3)
!2 = !DIFile(filename: "foo.h", directory: "/tmp")
!3 = !DIDerivedType(tag: DW_TAG_typedef, name: "__u64_t", file: !0, baseType: !4)
!4 = !DIBasicType(name: "unsigned long long", size: 64, encoding: DW_ATE_unsigned)
!5 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "My Compiler", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !6, retainedTypes: !6, globals: !7)
!6 = !{}
!7 = !{}
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{!"My Compiler"}
!11 = distinct !DISubprogram(name: "func_5", scope: !0, file: !0, line: 117, type: !12, isLocal: true, isDefinition: true, scopeLine: 118, isOptimized: false, unit: !5, variables: !6)
!12 = !DISubroutineType(types: !13)
!13 = !{}
!14 = !DILocalVariable(name: "p_6", arg: 1, scope: !11, line: 117, type: !1)
!15 = !DILocation(line: 117, column: 34, scope: !11)

; instcombine should realize that the load will read 42 from g and pass 42 to
; gaz regardless of the dbg.value in bar.

; CHECK: define void @foo() {
; CHECK-NEXT:  store i8 42, i8* @g, align 1
; CHECK-NEXT:  call void @bar(i8 1)
; CHECK-NEXT:  call void @gaz(i8 42)
; CHECK-NEXT:  ret void

