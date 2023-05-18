; RUN: opt -passes=inline < %s -S | FileCheck %s
; RUN: opt -passes=inline < %s -S --try-experimental-debuginfo-iterators | FileCheck %s

; CHECK: define {{.*}}@f1
; CHECK-NOT: define

%a = type { ptr, ptr }

$f3 = comdat any

define linkonce_odr void @f1() {
  call void @f2(ptr @f3)
  ret void
}

define linkonce_odr void @f2(ptr %__f) {
  call void @llvm.dbg.value(metadata ptr %__f, metadata !2, metadata !DIExpression()), !dbg !10
  call void %__f()
  ret void
}

define linkonce_odr void @f3() comdat {
  ret void
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.module.flags = !{!0, !1}

!0 = !{i32 7, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !DILocalVariable(name: "__f", arg: 3, scope: !3, file: !4, line: 3814, type: !9)
!3 = distinct !DISubprogram(scope: !5, file: !4, line: 3814, type: !6, scopeLine: 3815, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8, templateParams: !7, retainedNodes: !7)
!4 = !DIFile(filename: "a", directory: "")
!5 = !DINamespace(name: "std", scope: null)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !4, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !7, retainedTypes: !7, globals: !7, imports: !7, splitDebugInlining: false, nameTableKind: None)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64)
!10 = !DILocation(line: 0, scope: !3)
