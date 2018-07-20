; ModuleID = 'bar.cpp'

%class.B = type { i32, %class.A* }
%class.A = type { %class.Base, i32 }
%class.Base = type { i32 }

; Function Attrs: nounwind ssp uwtable
define void @_Z1gi(i32 %a) #0 !dbg !20 {
entry:
  %a.addr = alloca i32, align 4
  %t = alloca %class.B, align 8
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !28, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.declare(metadata %class.B* %t, metadata !30, metadata !DIExpression()), !dbg !31
  ret void, !dbg !32
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: ssp uwtable
define i32 @main() #2 !dbg !24 {
entry:
  %retval = alloca i32, align 4
  %a = alloca %class.A, align 4
  store i32 0, i32* %retval
  call void @llvm.dbg.declare(metadata %class.A* %a, metadata !33, metadata !DIExpression()), !dbg !34
  call void @_Z1fi(i32 0), !dbg !35
  call void @_Z1gi(i32 1), !dbg !36
  ret i32 0, !dbg !37
}

declare void @_Z1fi(i32) #3

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!27, !38}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.4 (http://llvm.org/git/clang.git f54e02f969d02d640103db73efc30c45439fceab) (http://llvm.org/git/llvm.git 284353b55896cb1babfaa7add7c0a363245342d2)", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "bar.cpp", directory: "/Users/mren/c_testing/type_unique_air/inher")
!2 = !{}
!3 = !{!4, !11, !15}
!4 = !DICompositeType(tag: DW_TAG_class_type, name: "B", line: 7, size: 128, align: 64, file: !5, elements: !6, identifier: "_ZTS1B")
!5 = !DIFile(filename: "./b.hpp", directory: "/Users/mren/c_testing/type_unique_air/inher")
!6 = !{!7, !9}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "bb", line: 8, size: 32, align: 32, flags: DIFlagPrivate, file: !5, scope: !4, baseType: !8)
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "a", line: 9, size: 64, align: 64, offset: 64, flags: DIFlagPrivate, file: !5, scope: !4, baseType: !10)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !11)
!11 = !DICompositeType(tag: DW_TAG_class_type, name: "A", line: 3, size: 64, align: 32, file: !12, elements: !13, identifier: "_ZTS1A")
!12 = !DIFile(filename: "./a.hpp", directory: "/Users/mren/c_testing/type_unique_air/inher")
!13 = !{!14, !18}
!14 = !DIDerivedType(tag: DW_TAG_inheritance, flags: DIFlagPrivate, scope: !11, baseType: !15)
!15 = !DICompositeType(tag: DW_TAG_class_type, name: "Base", line: 3, size: 32, align: 32, file: !5, elements: !16, identifier: "_ZTS4Base")
!16 = !{!17}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "b", line: 4, size: 32, align: 32, flags: DIFlagPrivate, file: !5, scope: !15, baseType: !8)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "x", line: 4, size: 32, align: 32, offset: 32, flags: DIFlagPrivate, file: !12, scope: !11, baseType: !8)
!20 = distinct !DISubprogram(name: "g", linkageName: "_Z1gi", line: 4, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 4, file: !1, scope: !21, type: !22, retainedNodes: !2)
!21 = !DIFile(filename: "bar.cpp", directory: "/Users/mren/c_testing/type_unique_air/inher")
!22 = !DISubroutineType(types: !23)
!23 = !{null, !8}
!24 = distinct !DISubprogram(name: "main", line: 9, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 9, file: !1, scope: !21, type: !25, retainedNodes: !2)
!25 = !DISubroutineType(types: !26)
!26 = !{!8}
!27 = !{i32 2, !"Dwarf Version", i32 2}
!28 = !DILocalVariable(name: "a", line: 4, arg: 1, scope: !20, file: !21, type: !8)
!29 = !DILocation(line: 4, scope: !20)
!30 = !DILocalVariable(name: "t", line: 5, scope: !20, file: !21, type: !4)
!31 = !DILocation(line: 5, scope: !20)
!32 = !DILocation(line: 6, scope: !20)
!33 = !DILocalVariable(name: "a", line: 10, scope: !24, file: !21, type: !11)
!34 = !DILocation(line: 10, scope: !24)
!35 = !DILocation(line: 11, scope: !24)
!36 = !DILocation(line: 12, scope: !24)
!37 = !DILocation(line: 13, scope: !24)
!38 = !{i32 1, !"Debug Info Version", i32 3}
