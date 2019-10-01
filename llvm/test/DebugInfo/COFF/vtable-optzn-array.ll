; RUN: llc < %s -filetype=obj -o - | llvm-readobj - --codeview | FileCheck %s

; We used to crash on this input because UnicodeString is a forward declaration
; with no size. Our array type logic wanted to assert the size of the elements
; times the length matched the overall size.

; C++ source to regenerate:
; struct UnicodeString;
; struct GetFwdDecl {
;   static UnicodeString format;
; };
; GetFwdDecl force_fwd_decl;
; struct UnicodeString {
; private:
;   virtual ~UnicodeString();
; };
; struct UseCompleteType {
;   UseCompleteType();
;   ~UseCompleteType();
;   UnicodeString currencySpcAfterSym[1];
; };
; UseCompleteType require_complete;

; CHECK:      Array ({{.*}}) {
; CHECK-NEXT:   TypeLeafKind: LF_ARRAY (0x1503)
; CHECK-NEXT:   ElementType: UnicodeString
; CHECK-NEXT:   IndexType: unsigned __int64 (0x23)
; CHECK-NEXT:   SizeOf: 8
; CHECK-NEXT:   Name:
; CHECK-NEXT: }

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

%struct.GetFwdDecl = type { i8 }
%struct.UseCompleteType = type { [1 x %struct.UnicodeString] }
%struct.UnicodeString = type { i32 (...)** }

$"\01??_DUseCompleteType@@QEAA@XZ" = comdat any

@"\01?force_fwd_decl@@3UGetFwdDecl@@A" = global %struct.GetFwdDecl zeroinitializer, align 1, !dbg !0
@"\01?require_complete@@3UUseCompleteType@@A" = global %struct.UseCompleteType zeroinitializer, align 8, !dbg !6
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_t.cpp, i8* null }]

; Function Attrs: nounwind
define internal void @"\01??__Erequire_complete@@YAXXZ"() #0 !dbg !27 {
entry:
  %call = call %struct.UseCompleteType* @"\01??0UseCompleteType@@QEAA@XZ"(%struct.UseCompleteType* @"\01?require_complete@@3UUseCompleteType@@A"), !dbg !30
  %0 = call i32 @atexit(void ()* @"\01??__Frequire_complete@@YAXXZ") #2, !dbg !30
  ret void, !dbg !30
}

declare %struct.UseCompleteType* @"\01??0UseCompleteType@@QEAA@XZ"(%struct.UseCompleteType* returned) unnamed_addr #1

; Function Attrs: nounwind
define linkonce_odr void @"\01??_DUseCompleteType@@QEAA@XZ"(%struct.UseCompleteType* %this) unnamed_addr #0 comdat align 2 !dbg !31 {
entry:
  %this.addr = alloca %struct.UseCompleteType*, align 8
  store %struct.UseCompleteType* %this, %struct.UseCompleteType** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.UseCompleteType** %this.addr, metadata !32, metadata !34), !dbg !35
  %this1 = load %struct.UseCompleteType*, %struct.UseCompleteType** %this.addr, align 8
  call void @"\01??1UseCompleteType@@QEAA@XZ"(%struct.UseCompleteType* %this1), !dbg !36
  ret void, !dbg !36
}

; Function Attrs: nounwind

define internal void @"\01??__Frequire_complete@@YAXXZ"() #0 !dbg !37 {
entry:
  call void @"\01??_DUseCompleteType@@QEAA@XZ"(%struct.UseCompleteType* @"\01?require_complete@@3UUseCompleteType@@A"), !dbg !38
  ret void, !dbg !39
}

; Function Attrs: nounwind
declare i32 @atexit(void ()*) #2

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #3

declare void @"\01??1UseCompleteType@@QEAA@XZ"(%struct.UseCompleteType*) unnamed_addr #1

; Function Attrs: nounwind
define internal void @_GLOBAL__sub_I_t.cpp() #0 !dbg !41 {
entry:
  call void @"\01??__Erequire_complete@@YAXXZ"(), !dbg !43
  ret void
}

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { nounwind readnone }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!24, !25}
!llvm.ident = !{!26}

!0 = distinct !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "force_fwd_decl", linkageName: "\01?force_fwd_decl@@3UGetFwdDecl@@A", scope: !2, file: !8, line: 5, type: !21, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 4.0.0 (trunk 281056) (llvm/trunk 281051)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "<stdin>", directory: "C:\5Csrc\5Cllvm\5Cbuild")
!4 = !{}
!5 = !{!0, !6}
!6 = distinct !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = !DIGlobalVariable(name: "require_complete", linkageName: "\01?require_complete@@3UUseCompleteType@@A", scope: !2, file: !8, line: 15, type: !9, isLocal: false, isDefinition: true)
!8 = !DIFile(filename: "t.cpp", directory: "C:\5Csrc\5Cllvm\5Cbuild")
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "UseCompleteType", file: !8, line: 10, size: 64, align: 64, elements: !10, identifier: ".?AUUseCompleteType@@")
!10 = !{!11, !16, !20}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "currencySpcAfterSym", scope: !9, file: !8, line: 13, baseType: !12, size: 64, align: 64)
!12 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 64, align: 64, elements: !14)
!13 = !DICompositeType(tag: DW_TAG_structure_type, name: "UnicodeString", file: !8, line: 1, flags: DIFlagFwdDecl, identifier: ".?AUUnicodeString@@")
!14 = !{!15}
!15 = !DISubrange(count: 1)
!16 = !DISubprogram(name: "UseCompleteType", scope: !9, file: !8, line: 11, type: !17, isLocal: false, isDefinition: false, scopeLine: 11, flags: DIFlagPrototyped, isOptimized: false)
!17 = !DISubroutineType(types: !18)
!18 = !{null, !19}
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!20 = !DISubprogram(name: "~UseCompleteType", scope: !9, file: !8, line: 12, type: !17, isLocal: false, isDefinition: false, scopeLine: 12, flags: DIFlagPrototyped, isOptimized: false)
!21 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "GetFwdDecl", file: !8, line: 2, size: 8, align: 8, elements: !22, identifier: ".?AUGetFwdDecl@@")
!22 = !{!23}
!23 = !DIDerivedType(tag: DW_TAG_member, name: "format", scope: !21, file: !8, line: 3, baseType: !13, flags: DIFlagStaticMember)
!24 = !{i32 2, !"CodeView", i32 1}
!25 = !{i32 2, !"Debug Info Version", i32 3}
!26 = !{!"clang version 4.0.0 (trunk 281056) (llvm/trunk 281051)"}
!27 = distinct !DISubprogram(name: "??__Erequire_complete@@YAXXZ", scope: !8, file: !8, line: 15, type: !28, isLocal: true, isDefinition: true, scopeLine: 15, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!28 = !DISubroutineType(types: !29)
!29 = !{null}
!30 = !DILocation(line: 15, scope: !27)
!31 = distinct !DISubprogram(name: "~UseCompleteType", linkageName: "\01??_DUseCompleteType@@QEAA@XZ", scope: !9, file: !8, line: 12, type: !17, isLocal: false, isDefinition: true, scopeLine: 15, flags: DIFlagPrototyped, isOptimized: false, unit: !2, declaration: !20, retainedNodes: !4)
!32 = !DILocalVariable(name: "this", arg: 1, scope: !31, type: !33, flags: DIFlagArtificial | DIFlagObjectPointer)
!33 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64, align: 64)
!34 = !DIExpression()
!35 = !DILocation(line: 0, scope: !31)
!36 = !DILocation(line: 15, scope: !31)
!37 = distinct !DISubprogram(name: "??__Frequire_complete@@YAXXZ", scope: !3, file: !3, line: 15, type: !28, isLocal: true, isDefinition: true, scopeLine: 15, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!38 = !DILocation(line: 15, scope: !37)
!39 = !DILocation(line: 15, scope: !40)
!40 = !DILexicalBlockFile(scope: !37, file: !8, discriminator: 0)
!41 = distinct !DISubprogram(linkageName: "_GLOBAL__sub_I_t.cpp", scope: !3, file: !3, type: !42, isLocal: true, isDefinition: true, flags: DIFlagArtificial, isOptimized: false, unit: !2, retainedNodes: !4)
!42 = !DISubroutineType(types: !4)
!43 = !DILocation(line: 0, scope: !41)

