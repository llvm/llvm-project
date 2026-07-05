; RUN: llc -mtriple=x86_64-apple-macosx10.7 %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s

; CHECK: b_ref
; CHECK-NOT: AT_bit_size

%struct.bar = type { %struct.baz, ptr }
%struct.baz = type { i32 }

define i32 @main(i32 %argc, ptr %argv) uwtable ssp !dbg !29 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca ptr, align 8
  %myBar = alloca %struct.bar, align 8
  store i32 0, ptr %retval
  store i32 %argc, ptr %argc.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %argc.addr, metadata !49, metadata !DIExpression()), !dbg !50
  store ptr %argv, ptr %argv.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %argv.addr, metadata !51, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.declare(metadata ptr %myBar, metadata !53, metadata !DIExpression()), !dbg !55
  call void @_ZN3barC1Ei(ptr %myBar, i32 1), !dbg !56
  ret i32 0, !dbg !57
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define linkonce_odr void @_ZN3barC1Ei(ptr %this, i32 %x) unnamed_addr uwtable ssp align 2 !dbg !37 {
entry:
  %this.addr = alloca ptr, align 8
  %x.addr = alloca i32, align 4
  store ptr %this, ptr %this.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %this.addr, metadata !58, metadata !DIExpression()), !dbg !59
  store i32 %x, ptr %x.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %x.addr, metadata !60, metadata !DIExpression()), !dbg !61
  %this1 = load ptr, ptr %this.addr
  %0 = load i32, ptr %x.addr, align 4, !dbg !62
  call void @_ZN3barC2Ei(ptr %this1, i32 %0), !dbg !62
  ret void, !dbg !62
}

define linkonce_odr void @_ZN3barC2Ei(ptr %this, i32 %x) unnamed_addr uwtable ssp align 2 !dbg !40 {
entry:
  %this.addr = alloca ptr, align 8
  %x.addr = alloca i32, align 4
  store ptr %this, ptr %this.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %this.addr, metadata !63, metadata !DIExpression()), !dbg !64
  store i32 %x, ptr %x.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %x.addr, metadata !65, metadata !DIExpression()), !dbg !66
  %this1 = load ptr, ptr %this.addr
  %0 = load i32, ptr %x.addr, align 4, !dbg !67
  call void @_ZN3bazC1Ei(ptr %this1, i32 %0), !dbg !67
  %1 = getelementptr inbounds %struct.bar, ptr %this1, i32 0, i32 1, !dbg !67
  store ptr %this1, ptr %1, align 8, !dbg !67
  ret void, !dbg !68
}

define linkonce_odr void @_ZN3bazC1Ei(ptr %this, i32 %a) unnamed_addr uwtable ssp align 2 !dbg !43 {
entry:
  %this.addr = alloca ptr, align 8
  %a.addr = alloca i32, align 4
  store ptr %this, ptr %this.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %this.addr, metadata !70, metadata !DIExpression()), !dbg !71
  store i32 %a, ptr %a.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %a.addr, metadata !72, metadata !DIExpression()), !dbg !73
  %this1 = load ptr, ptr %this.addr
  %0 = load i32, ptr %a.addr, align 4, !dbg !74
  call void @_ZN3bazC2Ei(ptr %this1, i32 %0), !dbg !74
  ret void, !dbg !74
}

define linkonce_odr void @_ZN3bazC2Ei(ptr %this, i32 %a) unnamed_addr nounwind uwtable ssp align 2 !dbg !46 {
entry:
  %this.addr = alloca ptr, align 8
  %a.addr = alloca i32, align 4
  store ptr %this, ptr %this.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %this.addr, metadata !75, metadata !DIExpression()), !dbg !76
  store i32 %a, ptr %a.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %a.addr, metadata !77, metadata !DIExpression()), !dbg !78
  %this1 = load ptr, ptr %this.addr
  %0 = load i32, ptr %a.addr, align 4, !dbg !79
  store i32 %0, ptr %this1, align 4, !dbg !79
  ret void, !dbg !80
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!83}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.1 (trunk 146596)", isOptimized: false, emissionKind: FullDebug, file: !82, enums: !1, retainedTypes: !3, globals: !1, imports:  !1)
!1 = !{}
!3 = !{!5, !9}
!5 = !DICompositeType(tag: DW_TAG_class_type, name: "bar", line: 9, size: 128, align: 64, file: !82, elements: !7)
!6 = !DIFile(filename: "main.cpp", directory: "/Users/echristo/tmp/bad-struct-ref")
!7 = !{!8, !19, !21}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "b", line: 11, size: 32, align: 32, file: !82, scope: !5, baseType: !9)
!9 = !DICompositeType(tag: DW_TAG_class_type, name: "baz", line: 3, size: 32, align: 32, file: !82, elements: !10)
!10 = !{!11, !13}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "h", line: 5, size: 32, align: 32, file: !82, scope: !9, baseType: !12)
!12 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!13 = !DISubprogram(name: "baz", line: 6, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, file: !82, scope: !9, type: !14)
!14 = !DISubroutineType(types: !15)
!15 = !{null, !16, !12}
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial, baseType: !9)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "b_ref", line: 12, size: 64, align: 64, offset: 64, file: !82, scope: !5, baseType: !20)
!20 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !9)
!21 = !DISubprogram(name: "bar", line: 13, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, file: !82, scope: !5, type: !22)
!22 = !DISubroutineType(types: !23)
!23 = !{null, !24, !12}
!24 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial, baseType: !5)
!29 = distinct !DISubprogram(name: "main", line: 17, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, file: !82, scope: !6, type: !30)
!30 = !DISubroutineType(types: !31)
!31 = !{!12, !12, !32}
!32 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !33)
!33 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !34)
!34 = !DIBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!35 = !{!36}
!36 = !{} ; previously: invalid DW_TAG_base_type
!37 = distinct !DISubprogram(name: "bar", linkageName: "_ZN3barC1Ei", line: 13, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, file: !82, scope: null, type: !22, declaration: !21)
!38 = !{!39}
!39 = !{} ; previously: invalid DW_TAG_base_type
!40 = distinct !DISubprogram(name: "bar", linkageName: "_ZN3barC2Ei", line: 13, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, file: !82, scope: null, type: !22, declaration: !21)
!41 = !{!42}
!42 = !{} ; previously: invalid DW_TAG_base_type
!43 = distinct !DISubprogram(name: "baz", linkageName: "_ZN3bazC1Ei", line: 6, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, file: !82, scope: null, type: !14, declaration: !13)
!44 = !{!45}
!45 = !{} ; previously: invalid DW_TAG_base_type
!46 = distinct !DISubprogram(name: "baz", linkageName: "_ZN3bazC2Ei", line: 6, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, file: !82, scope: null, type: !14, declaration: !13)
!49 = !DILocalVariable(name: "argc", line: 16, arg: 1, scope: !29, file: !6, type: !12)
!50 = !DILocation(line: 16, column: 14, scope: !29)
!51 = !DILocalVariable(name: "argv", line: 16, arg: 2, scope: !29, file: !6, type: !32)
!52 = !DILocation(line: 16, column: 27, scope: !29)
!53 = !DILocalVariable(name: "myBar", line: 18, scope: !54, file: !6, type: !5)
!54 = distinct !DILexicalBlock(line: 17, column: 1, file: !82, scope: !29)
!55 = !DILocation(line: 18, column: 9, scope: !54)
!56 = !DILocation(line: 18, column: 17, scope: !54)
!57 = !DILocation(line: 19, column: 5, scope: !54)
!58 = !DILocalVariable(name: "this", line: 13, arg: 1, flags: DIFlagArtificial, scope: !37, file: !6, type: !24)
!59 = !DILocation(line: 13, column: 5, scope: !37)
!60 = !DILocalVariable(name: "x", line: 13, arg: 2, scope: !37, file: !6, type: !12)
!61 = !DILocation(line: 13, column: 13, scope: !37)
!62 = !DILocation(line: 13, column: 34, scope: !37)
!63 = !DILocalVariable(name: "this", line: 13, arg: 1, flags: DIFlagArtificial, scope: !40, file: !6, type: !24)
!64 = !DILocation(line: 13, column: 5, scope: !40)
!65 = !DILocalVariable(name: "x", line: 13, arg: 2, scope: !40, file: !6, type: !12)
!66 = !DILocation(line: 13, column: 13, scope: !40)
!67 = !DILocation(line: 13, column: 33, scope: !40)
!68 = !DILocation(line: 13, column: 34, scope: !69)
!69 = distinct !DILexicalBlock(line: 13, column: 33, file: !82, scope: !40)
!70 = !DILocalVariable(name: "this", line: 6, arg: 1, flags: DIFlagArtificial, scope: !43, file: !6, type: !16)
!71 = !DILocation(line: 6, column: 5, scope: !43)
!72 = !DILocalVariable(name: "a", line: 6, arg: 2, scope: !43, file: !6, type: !12)
!73 = !DILocation(line: 6, column: 13, scope: !43)
!74 = !DILocation(line: 6, column: 24, scope: !43)
!75 = !DILocalVariable(name: "this", line: 6, arg: 1, flags: DIFlagArtificial, scope: !46, file: !6, type: !16)
!76 = !DILocation(line: 6, column: 5, scope: !46)
!77 = !DILocalVariable(name: "a", line: 6, arg: 2, scope: !46, file: !6, type: !12)
!78 = !DILocation(line: 6, column: 13, scope: !46)
!79 = !DILocation(line: 6, column: 23, scope: !46)
!80 = !DILocation(line: 6, column: 24, scope: !81)
!81 = distinct !DILexicalBlock(line: 6, column: 23, file: !82, scope: !46)
!82 = !DIFile(filename: "main.cpp", directory: "/Users/echristo/tmp/bad-struct-ref")
!83 = !{i32 1, !"Debug Info Version", i32 3}
