; ModuleID = 'bar.cpp'

%struct.Base = type { i32, %struct.Base* }

; Function Attrs: nounwind ssp uwtable
define void @_Z1gi(i32 %a) #0 !dbg !12 {
entry:
  %a.addr = alloca i32, align 4
  %t = alloca %struct.Base, align 8
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !20, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.declare(metadata %struct.Base* %t, metadata !22, metadata !DIExpression()), !dbg !23
  ret void, !dbg !24
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: ssp uwtable
define i32 @main() #2 !dbg !16 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  call void @_Z1fi(i32 0), !dbg !25
  call void @_Z1gi(i32 1), !dbg !26
  ret i32 0, !dbg !27
}

declare void @_Z1fi(i32) #3

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!19, !28}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.4", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "bar.cpp", directory: ".")
!2 = !{}
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "Base", line: 1, file: !5, elements: !6, identifier: "_ZTS4Base")
!5 = !DIFile(filename: "./a.hpp", directory: ".")
!6 = !{!7, !9}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "a", line: 2, size: 32, align: 32, file: !5, scope: !4, baseType: !8)
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "b", line: 3, size: 64, align: 64, offset: 64, file: !5, scope: !4, baseType: !10)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4)
!12 = distinct !DISubprogram(name: "g", linkageName: "_Z1gi", line: 4, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 4, file: !1, scope: !13, type: !14, retainedNodes: !2)
!13 = !DIFile(filename: "bar.cpp", directory: ".")
!14 = !DISubroutineType(types: !15)
!15 = !{null, !8}
!16 = distinct !DISubprogram(name: "main", line: 7, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 7, file: !1, scope: !13, type: !17, retainedNodes: !2)
!17 = !DISubroutineType(types: !18)
!18 = !{!8}
!19 = !{i32 2, !"Dwarf Version", i32 2}
!20 = !DILocalVariable(name: "a", line: 4, arg: 1, scope: !12, file: !13, type: !8)
!21 = !DILocation(line: 4, scope: !12)
!22 = !DILocalVariable(name: "t", line: 5, scope: !12, file: !13, type: !4)
!23 = !DILocation(line: 5, scope: !12)
!24 = !DILocation(line: 6, scope: !12)
!25 = !DILocation(line: 8, scope: !16)
!26 = !DILocation(line: 9, scope: !16)
!27 = !DILocation(line: 10, scope: !16)
!28 = !{i32 1, !"Debug Info Version", i32 3}
