
; RUN: llc -filetype=obj -O0 -generate-type-units -mtriple=x86_64-unknown-linux-gnu < %s \
; RUN:     | llvm-dwarfdump -debug-info -debug-types - | FileCheck %s

; CHECK: Compile Unit:

; CHECK: Type Unit:
; CHECK:   DW_TAG_structure
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:     DW_AT_name      ("foo")
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:     DW_TAG_member
; CHECK-NEXT:       DW_AT_name    ("x")
; CHECK-NEXT:       DW_AT_type    ({{.*}} "int [1]"

; But make sure we still use a type unit for an anonymous type that still has a
; name for linkage purposes (due to being defined in a typedef).

; CHECK: Type Unit:
; CHECK:   DW_TAG_structure
; CHECK-NEXT:     DW_AT_calling_convention
; CHECK-NEXT:     DW_AT_byte_size
; CHECK-NEXT:     DW_AT_decl_file
; CHECK-NEXT:     DW_AT_decl_line
; CHECK-NOT: DW
; CHECK:   NULL

%struct.foo = type { [1 x i32] }
%struct.bar = type { i8 }

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @_Z1f3foo3bar(i32 %.coerce) #0 !dbg !7 {
entry:
  %0 = alloca %struct.foo, align 4
  %1 = alloca %struct.bar, align 1
  %coerce.dive = getelementptr inbounds %struct.foo, %struct.foo* %0, i32 0, i32 0
  %2 = bitcast [1 x i32]* %coerce.dive to i32*
  store i32 %.coerce, i32* %2, align 4
  call void @llvm.dbg.declare(metadata %struct.foo* %0, metadata !19, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.declare(metadata %struct.bar* %1, metadata !21, metadata !DIExpression()), !dbg !22
  ret void, !dbg !23
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 9.0.0 (trunk 360374) (llvm/trunk 360380)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "named_types.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 9.0.0 (trunk 360374) (llvm/trunk 360380)"}
!7 = distinct !DISubprogram(name: "f", linkageName: "_Z1f3foo3bar", scope: !1, file: !1, line: 6, type: !8, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10, !17}
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "foo", file: !1, line: 1, size: 32, flags: DIFlagTypePassByValue, elements: !11, identifier: "_ZTS3foo")
!11 = !{!12}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !10, file: !1, line: 2, baseType: !13, size: 32)
!13 = !DICompositeType(tag: DW_TAG_array_type, baseType: !14, size: 32, elements: !15)
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !{!16}
!16 = !DISubrange(count: 1)
!17 = !DIDerivedType(tag: DW_TAG_typedef, name: "bar", file: !1, line: 5, baseType: !18)
!18 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !1, line: 4, size: 8, flags: DIFlagTypePassByValue, elements: !2, identifier: "_ZTS3bar")
!19 = !DILocalVariable(arg: 1, scope: !7, file: !1, line: 6, type: !10)
!20 = !DILocation(line: 6, column: 11, scope: !7)
!21 = !DILocalVariable(arg: 2, scope: !7, file: !1, line: 6, type: !17)
!22 = !DILocation(line: 6, column: 16, scope: !7)
!23 = !DILocation(line: 7, column: 1, scope: !7)

