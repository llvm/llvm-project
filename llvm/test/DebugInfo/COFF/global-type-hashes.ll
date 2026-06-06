; RUN: llc -filetype=obj < %s > %t.obj
; RUN: obj2yaml %t.obj | FileCheck %s --check-prefix=YAML
; RUN: llc -filetype=asm < %s | FileCheck %s --check-prefix=ASM

; C++ source to regenerate:
; $ cat t.cpp
; struct Foo {
;   Foo(int x, int y) : X(x), Y(y) {}
;   int method() { return X + Y; }
;   int X;
;   int Y;
; };
; int main(int argc, char **argv) {
;   Foo F {argc, argc};
;   return F.method();
; };
; $ clang-cc1 -triple i686-pc-windows-msvc19.11.25547 -emit-llvm -gcodeview \
;   -debug-info-kind=limited -std=c++14 foo.cpp
;


; ModuleID = 'foo.cpp'
source_filename = "foo.cpp"
target datalayout = "e-m:x-p:32:32-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc19.11.25547"

%struct.Foo = type { i32, i32 }

$"??0Foo@@QAE@HH@Z" = comdat any

$"?method@Foo@@QAEHXZ" = comdat any

; Function Attrs: mustprogress noinline norecurse nounwind optnone
define dso_local noundef i32 @main(i32 noundef %argc, ptr noundef %argv) #0 !dbg !8 {
entry:
  %retval = alloca i32, align 4
  %argv.addr = alloca ptr, align 4
  %argc.addr = alloca i32, align 4
  %F = alloca %struct.Foo, align 4
  store i32 0, ptr %retval, align 4
  store ptr %argv, ptr %argv.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %argv.addr, metadata !17, metadata !DIExpression()), !dbg !18
  store i32 %argc, ptr %argc.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %argc.addr, metadata !19, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.declare(metadata ptr %F, metadata !21, metadata !DIExpression()), !dbg !33
  %0 = load i32, ptr %argc.addr, align 4, !dbg !34
  %1 = load i32, ptr %argc.addr, align 4, !dbg !35
  %call = call x86_thiscallcc noundef ptr @"??0Foo@@QAE@HH@Z"(ptr noundef nonnull align 4 dereferenceable(8) %F, i32 noundef %0, i32 noundef %1), !dbg !33
  %call1 = call x86_thiscallcc noundef i32 @"?method@Foo@@QAEHXZ"(ptr noundef nonnull align 4 dereferenceable(8) %F), !dbg !36
  ret i32 %call1, !dbg !37
}

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind optnone
define linkonce_odr dso_local x86_thiscallcc noundef ptr @"??0Foo@@QAE@HH@Z"(ptr noundef nonnull returned align 4 dereferenceable(8) %this, i32 noundef %x, i32 noundef %y) unnamed_addr #2 comdat align 2 !dbg !38 {
entry:
  %y.addr = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %this.addr = alloca ptr, align 4
  store i32 %y, ptr %y.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %y.addr, metadata !39, metadata !DIExpression()), !dbg !40
  store i32 %x, ptr %x.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %x.addr, metadata !41, metadata !DIExpression()), !dbg !42
  store ptr %this, ptr %this.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %this.addr, metadata !43, metadata !DIExpression()), !dbg !45
  %this1 = load ptr, ptr %this.addr, align 4
  %X = getelementptr inbounds %struct.Foo, ptr %this1, i32 0, i32 0, !dbg !46
  %0 = load i32, ptr %x.addr, align 4, !dbg !47
  store i32 %0, ptr %X, align 4, !dbg !46
  %Y = getelementptr inbounds %struct.Foo, ptr %this1, i32 0, i32 1, !dbg !48
  %1 = load i32, ptr %y.addr, align 4, !dbg !49
  store i32 %1, ptr %Y, align 4, !dbg !48
  ret ptr %this1, !dbg !50
}

; Function Attrs: mustprogress noinline nounwind optnone
define linkonce_odr dso_local x86_thiscallcc noundef i32 @"?method@Foo@@QAEHXZ"(ptr noundef nonnull align 4 dereferenceable(8) %this) #3 comdat align 2 !dbg !51 {
entry:
  %this.addr = alloca ptr, align 4
  store ptr %this, ptr %this.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %this.addr, metadata !52, metadata !DIExpression()), !dbg !53
  %this1 = load ptr, ptr %this.addr, align 4
  %X = getelementptr inbounds %struct.Foo, ptr %this1, i32 0, i32 0, !dbg !54
  %0 = load i32, ptr %X, align 4, !dbg !54
  %Y = getelementptr inbounds %struct.Foo, ptr %this1, i32 0, i32 1, !dbg !55
  %1 = load i32, ptr %Y, align 4, !dbg !55
  %add = add nsw i32 %0, %1, !dbg !56
  ret i32 %add, !dbg !57
}

attributes #0 = { mustprogress noinline norecurse nounwind optnone "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+x87" }
attributes #1 = { nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { noinline nounwind optnone "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+x87" }
attributes #3 = { mustprogress noinline nounwind optnone "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+x87" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 16.0.0 (https://github.com/llvm/llvm-project.git a784de783af5096e593c5e214c2c78215fe303f5)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "C:\\git\\llvm-project", checksumkind: CSK_MD5, checksum: "d54692241b2727e6ae75e9d429c51680")
!2 = !{i32 1, !"NumRegisterParameters", i32 0}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"CodeViewGHash", i32 1}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 1, !"wchar_size", i32 2}
!7 = !{!"clang version 16.0.0 (https://github.com/llvm/llvm-project.git a784de783af5096e593c5e214c2c78215fe303f5)"}
!8 = distinct !DISubprogram(name: "main", scope: !9, file: !9, line: 7, type: !10, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !16)
!9 = !DIFile(filename: "foo.cpp", directory: "C:\\git\\llvm-project", checksumkind: CSK_MD5, checksum: "d54692241b2727e6ae75e9d429c51680")
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !12, !13}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 32)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 32)
!15 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!16 = !{}
!17 = !DILocalVariable(name: "argv", arg: 2, scope: !8, file: !9, line: 7, type: !13)
!18 = !DILocation(line: 7, column: 27, scope: !8)
!19 = !DILocalVariable(name: "argc", arg: 1, scope: !8, file: !9, line: 7, type: !12)
!20 = !DILocation(line: 7, column: 14, scope: !8)
!21 = !DILocalVariable(name: "F", scope: !8, file: !9, line: 8, type: !22)
!22 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", file: !9, line: 1, size: 64, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !23, identifier: ".?AUFoo@@")
!23 = !{!24, !25, !26, !30}
!24 = !DIDerivedType(tag: DW_TAG_member, name: "X", scope: !22, file: !9, line: 4, baseType: !12, size: 32)
!25 = !DIDerivedType(tag: DW_TAG_member, name: "Y", scope: !22, file: !9, line: 5, baseType: !12, size: 32, offset: 32)
!26 = !DISubprogram(name: "Foo", scope: !22, file: !9, line: 2, type: !27, scopeLine: 2, flags: DIFlagPrototyped, spFlags: 0)
!27 = !DISubroutineType(cc: DW_CC_BORLAND_thiscall, types: !28)
!28 = !{null, !29, !12, !12}
!29 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !22, size: 32, flags: DIFlagArtificial | DIFlagObjectPointer)
!30 = !DISubprogram(name: "method", linkageName: "?method@Foo@@QAEHXZ", scope: !22, file: !9, line: 3, type: !31, scopeLine: 3, flags: DIFlagPrototyped, spFlags: 0)
!31 = !DISubroutineType(cc: DW_CC_BORLAND_thiscall, types: !32)
!32 = !{!12, !29}
!33 = !DILocation(line: 8, column: 7, scope: !8)
!34 = !DILocation(line: 8, column: 10, scope: !8)
!35 = !DILocation(line: 8, column: 16, scope: !8)
!36 = !DILocation(line: 9, column: 12, scope: !8)
!37 = !DILocation(line: 9, column: 3, scope: !8)
!38 = distinct !DISubprogram(name: "Foo", linkageName: "??0Foo@@QAE@HH@Z", scope: !22, file: !9, line: 2, type: !27, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !26, retainedNodes: !16)
!39 = !DILocalVariable(name: "y", arg: 3, scope: !38, file: !9, line: 2, type: !12)
!40 = !DILocation(line: 2, column: 18, scope: !38)
!41 = !DILocalVariable(name: "x", arg: 2, scope: !38, file: !9, line: 2, type: !12)
!42 = !DILocation(line: 2, column: 11, scope: !38)
!43 = !DILocalVariable(name: "this", arg: 1, scope: !38, type: !44, flags: DIFlagArtificial | DIFlagObjectPointer)
!44 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !22, size: 32)
!45 = !DILocation(line: 0, scope: !38)
!46 = !DILocation(line: 2, column: 23, scope: !38)
!47 = !DILocation(line: 2, column: 25, scope: !38)
!48 = !DILocation(line: 2, column: 29, scope: !38)
!49 = !DILocation(line: 2, column: 31, scope: !38)
!50 = !DILocation(line: 2, column: 35, scope: !38)
!51 = distinct !DISubprogram(name: "method", linkageName: "?method@Foo@@QAEHXZ", scope: !22, file: !9, line: 3, type: !31, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !30, retainedNodes: !16)
!52 = !DILocalVariable(name: "this", arg: 1, scope: !51, type: !44, flags: DIFlagArtificial | DIFlagObjectPointer)
!53 = !DILocation(line: 0, scope: !51)
!54 = !DILocation(line: 3, column: 25, scope: !51)
!55 = !DILocation(line: 3, column: 29, scope: !51)
!56 = !DILocation(line: 3, column: 27, scope: !51)
!57 = !DILocation(line: 3, column: 18, scope: !51)


; YAML: --- !COFF
; YAML: header:
; YAML:   Machine:         IMAGE_FILE_MACHINE_I386
; YAML:   Characteristics: [  ]
; YAML: sections:
; YAML:   - Name:            '.debug$T'
; YAML:     Characteristics: [ IMAGE_SCN_CNT_INITIALIZED_DATA, IMAGE_SCN_MEM_DISCARDABLE, IMAGE_SCN_MEM_READ ]
; YAML:     Alignment:       4
; YAML:     Types:
; YAML:       - Kind:            LF_POINTER
; YAML:         Pointer:
; YAML:           ReferentType:    1136
; YAML:           Attrs:           32778
; YAML:       - Kind:            LF_ARGLIST
; YAML:         ArgList:
; YAML:           ArgIndices:      [ 116, 4096 ]
; YAML:       - Kind:            LF_PROCEDURE
; YAML:         Procedure:
; YAML:           ReturnType:      116
; YAML:           CallConv:        NearC
; YAML:           Options:         [ None ]
; YAML:           ParameterCount:  2
; YAML:           ArgumentList:    4097
; YAML:       - Kind:            LF_FUNC_ID
; YAML:         FuncId:
; YAML:           ParentScope:     0
; YAML:           FunctionType:    4098
; YAML:           Name:            main
; YAML:       - Kind:            LF_STRUCTURE
; YAML:         Class:
; YAML:           MemberCount:     0
; YAML:           Options:         [ None, ForwardReference, HasUniqueName ]
; YAML:           FieldList:       0
; YAML:           Name:            Foo
; YAML:           UniqueName:      '.?AUFoo@@'
; YAML:           DerivationList:  0
; YAML:           VTableShape:     0
; YAML:           Size:            0
; YAML:       - Kind:            LF_POINTER
; YAML:         Pointer:
; YAML:           ReferentType:    4100
; YAML:           Attrs:           33802
; YAML:       - Kind:            LF_ARGLIST
; YAML:         ArgList:
; YAML:           ArgIndices:      [ 116, 116 ]
; YAML:       - Kind:            LF_MFUNCTION
; YAML:         MemberFunction:
; YAML:           ReturnType:      3
; YAML:           ClassType:       4100
; YAML:           ThisType:        4101
; YAML:           CallConv:        ThisCall
; YAML:           Options:         [ None, Constructor ]
; YAML:           ParameterCount:  2
; YAML:           ArgumentList:    4102
; YAML:           ThisPointerAdjustment: 0
; YAML:       - Kind:            LF_ARGLIST
; YAML:         ArgList:
; YAML:           ArgIndices:      [  ]
; YAML:       - Kind:            LF_MFUNCTION
; YAML:         MemberFunction:
; YAML:           ReturnType:      116
; YAML:           ClassType:       4100
; YAML:           ThisType:        4101
; YAML:           CallConv:        ThisCall
; YAML:           Options:         [ None ]
; YAML:           ParameterCount:  0
; YAML:           ArgumentList:    4104
; YAML:           ThisPointerAdjustment: 0
; YAML:       - Kind:            LF_FIELDLIST
; YAML:         FieldList:
; YAML:           - Kind:            LF_MEMBER
; YAML:             DataMember:
; YAML:               Attrs:           3
; YAML:               Type:            116
; YAML:               FieldOffset:     0
; YAML:               Name:            X
; YAML:           - Kind:            LF_MEMBER
; YAML:             DataMember:
; YAML:               Attrs:           3
; YAML:               Type:            116
; YAML:               FieldOffset:     4
; YAML:               Name:            Y
; YAML:           - Kind:            LF_ONEMETHOD
; YAML:             OneMethod:
; YAML:               Type:            4103
; YAML:               Attrs:           3
; YAML:               VFTableOffset:   -1
; YAML:               Name:            Foo
; YAML:           - Kind:            LF_ONEMETHOD
; YAML:             OneMethod:
; YAML:               Type:            4105
; YAML:               Attrs:           3
; YAML:               VFTableOffset:   -1
; YAML:               Name:            method
; YAML:       - Kind:            LF_STRUCTURE
; YAML:         Class:
; YAML:           MemberCount:     4
; YAML:           Options:         [ None, HasConstructorOrDestructor, HasUniqueName ]
; YAML:           FieldList:       4106
; YAML:           Name:            Foo
; YAML:           UniqueName:      '.?AUFoo@@'
; YAML:           DerivationList:  0
; YAML:           VTableShape:     0
; YAML:           Size:            8
; YAML:       - Kind:            LF_STRING_ID
; YAML:         StringId:
; YAML:           Id:              0
; YAML:           String:          'C:\git\llvm-project\foo.cpp'
; YAML:       - Kind:            LF_UDT_SRC_LINE
; YAML:         UdtSourceLine:
; YAML:           UDT:             4107
; YAML:           SourceFile:      4108
; YAML:           LineNumber:      1
; YAML:       - Kind:            LF_MFUNC_ID
; YAML:         MemberFuncId:
; YAML:           ClassType:       4100
; YAML:           FunctionType:    4103
; YAML:           Name:            Foo
; YAML:       - Kind:            LF_POINTER
; YAML:         Pointer:
; YAML:           ReferentType:    4100
; YAML:           Attrs:           32778
; YAML:       - Kind:            LF_MFUNC_ID
; YAML:         MemberFuncId:
; YAML:           ClassType:       4100
; YAML:           FunctionType:    4105
; YAML:           Name:            method
; YAML:       - Kind:            LF_STRING_ID
; YAML:         StringId:
; YAML:           Id:              0
; YAML:           String:          'C:\git\llvm-project'
; YAML:       - Kind:            LF_STRING_ID
; YAML:         StringId:
; YAML:           Id:              0
; YAML:           String:          '<stdin>'
; YAML:       - Kind:            LF_STRING_ID
; YAML:         StringId:
; YAML:           Id:              0
; YAML:           String:          ''
; YAML:       - Kind:            LF_BUILDINFO
; YAML:         BuildInfo:
; YAML:           ArgIndices:      [ 4113, 4115, 4114, 4115, 4115 ]
; YAML:   - Name:            '.debug$H'
; YAML:     Characteristics: [ IMAGE_SCN_CNT_INITIALIZED_DATA, IMAGE_SCN_MEM_DISCARDABLE, IMAGE_SCN_MEM_READ ]
; YAML:     Alignment:       4
; YAML:     GlobalHashes:
; YAML:       Version:         0
; YAML:       HashAlgorithm:   2
; YAML:       HashValues:
; YAML:         - 0FDF2CE06172DBE8
; YAML:         - B5E1C8329B9F4E7F
; YAML:         - 1EE1398011AA4BE1
; YAML:         - B682FB0B006CEC2E
; YAML:         - 8F2D2AE45F6E79E8
; YAML:         - 1747FDF05D25DDEE
; YAML:         - EAA738703837EBAE
; YAML:         - 07B9EF65EBA94121
; YAML:         - AFF81B6AE460D908
; YAML:         - 90DFD798AF84402C
; YAML:         - B9DDCF9F86BABE9E
; YAML:         - D1E2E5CAA3B96825
; YAML:         - 10994F943B4E46F3
; YAML:         - 4E2B6BC0E79F4271
; YAML:         - 72A4762DBB2AF2E4
; YAML:         - 1891CC40E9028AE7
; YAML:         - 1E6104ECC17E43DE
; YAML:         - 174CF4A3F5448049
; YAML:         - 5349856AF14E2246
; YAML:         - 55A48E0466FDCDA6
; YAML:         - EE6329A02D9F4959

; ASM:      .section        .debug$H,"dr"
; ASM-NEXT: .p2align        2
; ASM-NEXT: .long   20171205                # Magic
; ASM-NEXT: .short  0                       # Section Version
; ASM-NEXT: .short  2                       # Hash Algorithm
; ASM-NEXT: .byte   0x0f, 0xdf, 0x2c, 0xe0  # 0x1000 [0FDF2CE06172DBE8]
; ASM-NEXT: .byte   0x61, 0x72, 0xdb, 0xe8
; ASM-NEXT: .byte   0xb5, 0xe1, 0xc8, 0x32  # 0x1001 [B5E1C8329B9F4E7F]
; ASM-NEXT: .byte   0x9b, 0x9f, 0x4e, 0x7f
; ASM-NEXT: .byte   0x1e, 0xe1, 0x39, 0x80  # 0x1002 [1EE1398011AA4BE1]