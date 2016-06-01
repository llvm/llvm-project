; RUN: llc < %s -filetype=obj | llvm-readobj - -codeview | FileCheck %s

; C++ source to regenerate:
; $ cat t.cpp
; struct A {
;   int a;
;   void f();
; };
; void usevars(int, ...);
; void f(float p1, double p2, long long p3) {
;   int v1 = p3;
;   int *v2 = &v1;
;   const int *v21 = &v1;
;   void *v3 = &v1;
;   int A::*v4 = &A::a;
;   void (A::*v5)() = &A::f;
;   long l1 = 0;
;   long int l2 = 0;
;   unsigned long l3 = 0;
;   unsigned long int l4 = 0;
;   usevars(v1, v2, v3, l1, l2, l3, l4);
; }
; void CharTypes() {
;   signed wchar_t w;
;   unsigned short us;
;   char c;
;   unsigned char uc;
;   signed char sc;
;   char16_t c16;
;   char32_t c32;
; }
; $ clang t.cpp -S -emit-llvm -g -gcodeview -o t.ll

; CHECK: CodeViewTypes [
; CHECK:   Modifier (0x1000) {
; CHECK:     TypeLeafKind: LF_MODIFIER (0x1001)
; CHECK:     ModifiedType: int (0x74)
; CHECK:     Modifiers [ (0x1)
; CHECK:       Const (0x1)
; CHECK:     ]
; CHECK:   }
; CHECK:   Pointer (0x1001) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: const int (0x1000)
; CHECK:     PointerAttributes: 0x1000C
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: Pointer (0x0)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:   }
; CHECK:   Pointer (0x1002) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: int (0x74)
; CHECK:     PointerAttributes: 0x804C
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: PointerToDataMember (0x2)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     ClassType: 0x0
; CHECK:     Representation: Unknown (0x0)
; CHECK:   }
; CHECK:   Pointer (0x1003) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: 0x0
; CHECK:     PointerAttributes: 0x1006C
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: PointerToMemberFunction (0x3)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     ClassType: 0x0
; CHECK:     Representation: Unknown (0x0)
; CHECK:   }
; CHECK: ]
; CHECK: CodeViewDebugInfo [
; CHECK:   Subsection [
; CHECK:     SubSectionType: Symbols (0xF1)
; CHECK:     ProcStart {
; CHECK:       DbgStart: 0x0
; CHECK:       DbgEnd: 0x0
; CHECK:       FunctionType: 0x0
; CHECK:       CodeOffset: ?f@@YAXMN_J@Z+0x0
; CHECK:       Segment: 0x0
; CHECK:       Flags [ (0x0)
; CHECK:       ]
; CHECK:       DisplayName: f
; CHECK:       LinkageName: ?f@@YAXMN_J@Z
; CHECK:     }
; CHECK:     Local {
; CHECK:       Type: __int64 (0x76)
; CHECK:       Flags [ (0x1)
; CHECK:         IsParameter (0x1)
; CHECK:       ]
; CHECK:       VarName: p3
; CHECK:     }
; CHECK:     Local {
; CHECK:       Type: double (0x41)
; CHECK:       Flags [ (0x1)
; CHECK:         IsParameter (0x1)
; CHECK:       ]
; CHECK:       VarName: p2
; CHECK:     }
; CHECK:     Local {
; CHECK:       Type: float (0x40)
; CHECK:       Flags [ (0x1)
; CHECK:         IsParameter (0x1)
; CHECK:       ]
; CHECK:       VarName: p1
; CHECK:     }
; CHECK:     Local {
; CHECK:       Type: int (0x74)
; CHECK:       VarName: v1
; CHECK:     }
; CHECK:     Local {
; CHECK:       Type: int* (0x674)
; CHECK:       VarName: v2
; CHECK:     }
; CHECK:     Local {
; CHECK:       Type: const int* (0x1001)
; CHECK:       VarName: v21
; CHECK:     }
; CHECK:     Local {
; CHECK:       Type: void* (0x603)
; CHECK:       VarName: v3
; CHECK:     }
; CHECK:     Local {
; CHECK:       Type: int <no type>::* (0x1002)
; CHECK:       VarName: v4
; CHECK:     }
; CHECK:     Local {
; CHECK:       Type: <no type> <no type>::* (0x1003)
; CHECK:       VarName: v5
; CHECK:     }
; CHECK:     Local {
; CHECK:       Type: long (0x12)
; CHECK:       VarName: l1
; CHECK:     }
; CHECK:     Local {
; CHECK:       Type: long (0x12)
; CHECK:       VarName: l2
; CHECK:     }
; CHECK:     Local {
; CHECK:       Type: unsigned long (0x22)
; CHECK:       VarName: l3
; CHECK:     }
; CHECK:     Local {
; CHECK:       Type: unsigned long (0x22)
; CHECK:       VarName: l4
; CHECK:     }
; CHECK:     ProcEnd {
; CHECK:     }
; CHECK:   ]
; CHECK:   Subsection [
; CHECK:     ProcStart {
; CHECK:       DisplayName: CharTypes
; CHECK:       LinkageName: ?CharTypes@@YAXXZ
; CHECK:     }
; CHECK:     Local {
; CHECK:       Type: wchar_t (0x71)
; CHECK:       Flags [ (0x0)
; CHECK:       ]
; CHECK:       VarName: w
; CHECK:     }
; CHECK:     Local {
; CHECK:       Type: unsigned short (0x21)
; CHECK:       Flags [ (0x0)
; CHECK:       ]
; CHECK:       VarName: us
; CHECK:     }
; CHECK:     Local {
; CHECK:       Type: char (0x70)
; CHECK:       Flags [ (0x0)
; CHECK:       ]
; CHECK:       VarName: c
; CHECK:     }
; CHECK:     Local {
; CHECK:       Type: unsigned char (0x20)
; CHECK:       Flags [ (0x0)
; CHECK:       ]
; CHECK:       VarName: uc
; CHECK:     }
; CHECK:     Local {
; CHECK:       Type: signed char (0x10)
; CHECK:       Flags [ (0x0)
; CHECK:       ]
; CHECK:       VarName: sc
; CHECK:     }
; CHECK:     Local {
; CHECK:       Type: char16_t (0x7A)
; CHECK:       Flags [ (0x0)
; CHECK:       ]
; CHECK:       VarName: c16
; CHECK:     }
; CHECK:     Local {
; CHECK:       Type: char32_t (0x7B)
; CHECK:       Flags [ (0x0)
; CHECK:       ]
; CHECK:       VarName: c32
; CHECK:     }
; CHECK:     ProcEnd {
; CHECK:     }
; CHECK:   ]
; CHECK: ]

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.23918"

%struct.A = type { i32 }

; Function Attrs: uwtable
define void @"\01?f@@YAXMN_J@Z"(float %p1, double %p2, i64 %p3) #0 !dbg !7 {
entry:
  %p3.addr = alloca i64, align 8
  %p2.addr = alloca double, align 8
  %p1.addr = alloca float, align 4
  %v1 = alloca i32, align 4
  %v2 = alloca i32*, align 8
  %v21 = alloca i32*, align 8
  %v3 = alloca i8*, align 8
  %v4 = alloca i32, align 8
  %v5 = alloca i8*, align 8
  %l1 = alloca i32, align 4
  %l2 = alloca i32, align 4
  %l3 = alloca i32, align 4
  %l4 = alloca i32, align 4
  store i64 %p3, i64* %p3.addr, align 8
  call void @llvm.dbg.declare(metadata i64* %p3.addr, metadata !13, metadata !14), !dbg !15
  store double %p2, double* %p2.addr, align 8
  call void @llvm.dbg.declare(metadata double* %p2.addr, metadata !16, metadata !14), !dbg !17
  store float %p1, float* %p1.addr, align 4
  call void @llvm.dbg.declare(metadata float* %p1.addr, metadata !18, metadata !14), !dbg !19
  call void @llvm.dbg.declare(metadata i32* %v1, metadata !20, metadata !14), !dbg !22
  %0 = load i64, i64* %p3.addr, align 8, !dbg !23
  %conv = trunc i64 %0 to i32, !dbg !23
  store i32 %conv, i32* %v1, align 4, !dbg !22
  call void @llvm.dbg.declare(metadata i32** %v2, metadata !24, metadata !14), !dbg !26
  store i32* %v1, i32** %v2, align 8, !dbg !26
  call void @llvm.dbg.declare(metadata i32** %v21, metadata !27, metadata !14), !dbg !30
  store i32* %v1, i32** %v21, align 8, !dbg !30
  call void @llvm.dbg.declare(metadata i8** %v3, metadata !31, metadata !14), !dbg !33
  %1 = bitcast i32* %v1 to i8*, !dbg !34
  store i8* %1, i8** %v3, align 8, !dbg !33
  call void @llvm.dbg.declare(metadata i32* %v4, metadata !35, metadata !14), !dbg !44
  store i32 0, i32* %v4, align 8, !dbg !44
  call void @llvm.dbg.declare(metadata i8** %v5, metadata !45, metadata !14), !dbg !47
  store i8* bitcast (void (%struct.A*)* @"\01?f@A@@QEAAXXZ" to i8*), i8** %v5, align 8, !dbg !47
  call void @llvm.dbg.declare(metadata i32* %l1, metadata !48, metadata !14), !dbg !50
  store i32 0, i32* %l1, align 4, !dbg !50
  call void @llvm.dbg.declare(metadata i32* %l2, metadata !51, metadata !14), !dbg !52
  store i32 0, i32* %l2, align 4, !dbg !52
  call void @llvm.dbg.declare(metadata i32* %l3, metadata !53, metadata !14), !dbg !55
  store i32 0, i32* %l3, align 4, !dbg !55
  call void @llvm.dbg.declare(metadata i32* %l4, metadata !56, metadata !14), !dbg !57
  store i32 0, i32* %l4, align 4, !dbg !57
  %2 = load i32, i32* %l4, align 4, !dbg !58
  %3 = load i32, i32* %l3, align 4, !dbg !59
  %4 = load i32, i32* %l2, align 4, !dbg !60
  %5 = load i32, i32* %l1, align 4, !dbg !61
  %6 = load i8*, i8** %v3, align 8, !dbg !62
  %7 = load i32*, i32** %v2, align 8, !dbg !63
  %8 = load i32, i32* %v1, align 4, !dbg !64
  call void (i32, ...) @"\01?usevars@@YAXHZZ"(i32 %8, i32* %7, i8* %6, i32 %5, i32 %4, i32 %3, i32 %2), !dbg !65
  ret void, !dbg !66
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @"\01?f@A@@QEAAXXZ"(%struct.A*) #2

declare void @"\01?usevars@@YAXHZZ"(i32, ...) #2

; Function Attrs: nounwind uwtable
define void @"\01?CharTypes@@YAXXZ"() #3 !dbg !67 {
entry:
  %w = alloca i16, align 2
  %us = alloca i16, align 2
  %c = alloca i8, align 1
  %uc = alloca i8, align 1
  %sc = alloca i8, align 1
  %c16 = alloca i16, align 2
  %c32 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i16* %w, metadata !70, metadata !14), !dbg !72
  call void @llvm.dbg.declare(metadata i16* %us, metadata !73, metadata !14), !dbg !75
  call void @llvm.dbg.declare(metadata i8* %c, metadata !76, metadata !14), !dbg !78
  call void @llvm.dbg.declare(metadata i8* %uc, metadata !79, metadata !14), !dbg !81
  call void @llvm.dbg.declare(metadata i8* %sc, metadata !82, metadata !14), !dbg !84
  call void @llvm.dbg.declare(metadata i16* %c16, metadata !85, metadata !14), !dbg !87
  call void @llvm.dbg.declare(metadata i32* %c32, metadata !88, metadata !14), !dbg !90
  ret void, !dbg !91
}

attributes #0 = { uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 (trunk 271336) (llvm/trunk 271339)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"clang version 3.9.0 (trunk 271336) (llvm/trunk 271339)"}
!7 = distinct !DISubprogram(name: "f", linkageName: "\01?f@@YAXMN_J@Z", scope: !1, file: !1, line: 6, type: !8, isLocal: false, isDefinition: true, scopeLine: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10, !11, !12}
!10 = !DIBasicType(name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!11 = !DIBasicType(name: "double", size: 64, align: 64, encoding: DW_ATE_float)
!12 = !DIBasicType(name: "long long int", size: 64, align: 64, encoding: DW_ATE_signed)
!13 = !DILocalVariable(name: "p3", arg: 3, scope: !7, file: !1, line: 6, type: !12)
!14 = !DIExpression()
!15 = !DILocation(line: 6, column: 39, scope: !7)
!16 = !DILocalVariable(name: "p2", arg: 2, scope: !7, file: !1, line: 6, type: !11)
!17 = !DILocation(line: 6, column: 25, scope: !7)
!18 = !DILocalVariable(name: "p1", arg: 1, scope: !7, file: !1, line: 6, type: !10)
!19 = !DILocation(line: 6, column: 14, scope: !7)
!20 = !DILocalVariable(name: "v1", scope: !7, file: !1, line: 7, type: !21)
!21 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!22 = !DILocation(line: 7, column: 7, scope: !7)
!23 = !DILocation(line: 7, column: 12, scope: !7)
!24 = !DILocalVariable(name: "v2", scope: !7, file: !1, line: 8, type: !25)
!25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !21, size: 64, align: 64)
!26 = !DILocation(line: 8, column: 8, scope: !7)
!27 = !DILocalVariable(name: "v21", scope: !7, file: !1, line: 9, type: !28)
!28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !29, size: 64, align: 64)
!29 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !21)
!30 = !DILocation(line: 9, column: 14, scope: !7)
!31 = !DILocalVariable(name: "v3", scope: !7, file: !1, line: 10, type: !32)
!32 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64, align: 64)
!33 = !DILocation(line: 10, column: 9, scope: !7)
!34 = !DILocation(line: 10, column: 14, scope: !7)
!35 = !DILocalVariable(name: "v4", scope: !7, file: !1, line: 11, type: !36)
!36 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !21, size: 32, extraData: !37)
!37 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !1, line: 1, size: 32, align: 32, elements: !38)
!38 = !{!39, !40}
!39 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !37, file: !1, line: 2, baseType: !21, size: 32, align: 32)
!40 = !DISubprogram(name: "A::f", linkageName: "\01?f@A@@QEAAXXZ", scope: !37, file: !1, line: 3, type: !41, isLocal: false, isDefinition: false, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false)
!41 = !DISubroutineType(types: !42)
!42 = !{null, !43}
!43 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !37, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!44 = !DILocation(line: 11, column: 11, scope: !7)
!45 = !DILocalVariable(name: "v5", scope: !7, file: !1, line: 12, type: !46)
!46 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !41, size: 64, extraData: !37)
!47 = !DILocation(line: 12, column: 13, scope: !7)
!48 = !DILocalVariable(name: "l1", scope: !7, file: !1, line: 13, type: !49)
!49 = !DIBasicType(name: "long int", size: 32, align: 32, encoding: DW_ATE_signed)
!50 = !DILocation(line: 13, column: 8, scope: !7)
!51 = !DILocalVariable(name: "l2", scope: !7, file: !1, line: 14, type: !49)
!52 = !DILocation(line: 14, column: 12, scope: !7)
!53 = !DILocalVariable(name: "l3", scope: !7, file: !1, line: 15, type: !54)
!54 = !DIBasicType(name: "long unsigned int", size: 32, align: 32, encoding: DW_ATE_unsigned)
!55 = !DILocation(line: 15, column: 17, scope: !7)
!56 = !DILocalVariable(name: "l4", scope: !7, file: !1, line: 16, type: !54)
!57 = !DILocation(line: 16, column: 21, scope: !7)
!58 = !DILocation(line: 17, column: 35, scope: !7)
!59 = !DILocation(line: 17, column: 31, scope: !7)
!60 = !DILocation(line: 17, column: 27, scope: !7)
!61 = !DILocation(line: 17, column: 23, scope: !7)
!62 = !DILocation(line: 17, column: 19, scope: !7)
!63 = !DILocation(line: 17, column: 15, scope: !7)
!64 = !DILocation(line: 17, column: 11, scope: !7)
!65 = !DILocation(line: 17, column: 3, scope: !7)
!66 = !DILocation(line: 18, column: 1, scope: !7)
!67 = distinct !DISubprogram(name: "CharTypes", linkageName: "\01?CharTypes@@YAXXZ", scope: !1, file: !1, line: 19, type: !68, isLocal: false, isDefinition: true, scopeLine: 19, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!68 = !DISubroutineType(types: !69)
!69 = !{null}
!70 = !DILocalVariable(name: "w", scope: !67, file: !1, line: 20, type: !71)
!71 = !DIBasicType(name: "wchar_t", size: 16, align: 16, encoding: DW_ATE_unsigned)
!72 = !DILocation(line: 20, column: 18, scope: !67)
!73 = !DILocalVariable(name: "us", scope: !67, file: !1, line: 21, type: !74)
!74 = !DIBasicType(name: "unsigned short", size: 16, align: 16, encoding: DW_ATE_unsigned)
!75 = !DILocation(line: 21, column: 18, scope: !67)
!76 = !DILocalVariable(name: "c", scope: !67, file: !1, line: 22, type: !77)
!77 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!78 = !DILocation(line: 22, column: 8, scope: !67)
!79 = !DILocalVariable(name: "uc", scope: !67, file: !1, line: 23, type: !80)
!80 = !DIBasicType(name: "unsigned char", size: 8, align: 8, encoding: DW_ATE_unsigned_char)
!81 = !DILocation(line: 23, column: 17, scope: !67)
!82 = !DILocalVariable(name: "sc", scope: !67, file: !1, line: 24, type: !83)
!83 = !DIBasicType(name: "signed char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!84 = !DILocation(line: 24, column: 15, scope: !67)
!85 = !DILocalVariable(name: "c16", scope: !67, file: !1, line: 25, type: !86)
!86 = !DIBasicType(name: "char16_t", size: 16, align: 16, encoding: DW_ATE_UTF)
!87 = !DILocation(line: 25, column: 12, scope: !67)
!88 = !DILocalVariable(name: "c32", scope: !67, file: !1, line: 26, type: !89)
!89 = !DIBasicType(name: "char32_t", size: 32, align: 32, encoding: DW_ATE_UTF)
!90 = !DILocation(line: 26, column: 12, scope: !67)
!91 = !DILocation(line: 27, column: 1, scope: !67)
