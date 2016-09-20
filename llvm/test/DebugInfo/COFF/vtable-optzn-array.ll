; RUN: llc < %s -filetype=obj -o - | llvm-readobj - -codeview | FileCheck %s

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

@"\01?force_fwd_decl@@3UGetFwdDecl@@A" = global %struct.GetFwdDecl zeroinitializer, align 1, !dbg !4
@"\01?require_complete@@3UUseCompleteType@@A" = global %struct.UseCompleteType zeroinitializer, align 8, !dbg !10
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_t.cpp, i8* null }]

; Function Attrs: nounwind
define internal void @"\01??__Erequire_complete@@YAXXZ"() #0 !dbg !25 {
entry:
  %call = call %struct.UseCompleteType* @"\01??0UseCompleteType@@QEAA@XZ"(%struct.UseCompleteType* @"\01?require_complete@@3UUseCompleteType@@A"), !dbg !28
  %0 = call i32 @atexit(void ()* @"\01??__Frequire_complete@@YAXXZ") #2, !dbg !28
  ret void, !dbg !28
}

declare %struct.UseCompleteType* @"\01??0UseCompleteType@@QEAA@XZ"(%struct.UseCompleteType* returned) unnamed_addr #1

; Function Attrs: nounwind
define linkonce_odr void @"\01??_DUseCompleteType@@QEAA@XZ"(%struct.UseCompleteType* %this) unnamed_addr #0 comdat align 2 !dbg !29 {
entry:
  %this.addr = alloca %struct.UseCompleteType*, align 8
  store %struct.UseCompleteType* %this, %struct.UseCompleteType** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.UseCompleteType** %this.addr, metadata !30, metadata !32), !dbg !33
  %this1 = load %struct.UseCompleteType*, %struct.UseCompleteType** %this.addr, align 8
  call void @"\01??1UseCompleteType@@QEAA@XZ"(%struct.UseCompleteType* %this1), !dbg !34
  ret void, !dbg !34
}

; Function Attrs: nounwind
define internal void @"\01??__Frequire_complete@@YAXXZ"() #0 !dbg !35 {
entry:
  call void @"\01??_DUseCompleteType@@QEAA@XZ"(%struct.UseCompleteType* @"\01?require_complete@@3UUseCompleteType@@A"), !dbg !36
  ret void, !dbg !37
}

; Function Attrs: nounwind
declare i32 @atexit(void ()*) #2

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #3

declare void @"\01??1UseCompleteType@@QEAA@XZ"(%struct.UseCompleteType*) unnamed_addr #1

; Function Attrs: nounwind
define internal void @_GLOBAL__sub_I_t.cpp() #0 !dbg !39 {
entry:
  call void @"\01??__Erequire_complete@@YAXXZ"(), !dbg !41
  ret void
}

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22, !23}
!llvm.ident = !{!24}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 4.0.0 (trunk 281056) (llvm/trunk 281051)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3)
!1 = !DIFile(filename: "<stdin>", directory: "C:\5Csrc\5Cllvm\5Cbuild")
!2 = !{}
!3 = !{!4, !10}
!4 = distinct !DIGlobalVariable(name: "force_fwd_decl", linkageName: "\01?force_fwd_decl@@3UGetFwdDecl@@A", scope: !0, file: !5, line: 5, type: !6, isLocal: false, isDefinition: true)
!5 = !DIFile(filename: "t.cpp", directory: "C:\5Csrc\5Cllvm\5Cbuild")
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "GetFwdDecl", file: !5, line: 2, size: 8, align: 8, elements: !7, identifier: ".?AUGetFwdDecl@@")
!7 = !{!8}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "format", scope: !6, file: !5, line: 3, baseType: !9, flags: DIFlagStaticMember)
!9 = !DICompositeType(tag: DW_TAG_structure_type, name: "UnicodeString", file: !5, line: 1, flags: DIFlagFwdDecl, identifier: ".?AUUnicodeString@@")
!10 = distinct !DIGlobalVariable(name: "require_complete", linkageName: "\01?require_complete@@3UUseCompleteType@@A", scope: !0, file: !5, line: 15, type: !11, isLocal: false, isDefinition: true)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "UseCompleteType", file: !5, line: 10, size: 64, align: 64, elements: !12, identifier: ".?AUUseCompleteType@@")
!12 = !{!13, !17, !21}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "currencySpcAfterSym", scope: !11, file: !5, line: 13, baseType: !14, size: 64, align: 64)
!14 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 64, align: 64, elements: !15)
!15 = !{!16}
!16 = !DISubrange(count: 1)
!17 = !DISubprogram(name: "UseCompleteType", scope: !11, file: !5, line: 11, type: !18, isLocal: false, isDefinition: false, scopeLine: 11, flags: DIFlagPrototyped, isOptimized: false)
!18 = !DISubroutineType(types: !19)
!19 = !{null, !20}
!20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!21 = !DISubprogram(name: "~UseCompleteType", scope: !11, file: !5, line: 12, type: !18, isLocal: false, isDefinition: false, scopeLine: 12, flags: DIFlagPrototyped, isOptimized: false)
!22 = !{i32 2, !"CodeView", i32 1}
!23 = !{i32 2, !"Debug Info Version", i32 3}
!24 = !{!"clang version 4.0.0 (trunk 281056) (llvm/trunk 281051)"}
!25 = distinct !DISubprogram(name: "??__Erequire_complete@@YAXXZ", scope: !5, file: !5, line: 15, type: !26, isLocal: true, isDefinition: true, scopeLine: 15, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!26 = !DISubroutineType(types: !27)
!27 = !{null}
!28 = !DILocation(line: 15, scope: !25)
!29 = distinct !DISubprogram(name: "~UseCompleteType", linkageName: "\01??_DUseCompleteType@@QEAA@XZ", scope: !11, file: !5, line: 12, type: !18, isLocal: false, isDefinition: true, scopeLine: 15, flags: DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !21, variables: !2)
!30 = !DILocalVariable(name: "this", arg: 1, scope: !29, type: !31, flags: DIFlagArtificial | DIFlagObjectPointer)
!31 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64, align: 64)
!32 = !DIExpression()
!33 = !DILocation(line: 0, scope: !29)
!34 = !DILocation(line: 15, scope: !29)
!35 = distinct !DISubprogram(name: "??__Frequire_complete@@YAXXZ", scope: !1, file: !1, line: 15, type: !26, isLocal: true, isDefinition: true, scopeLine: 15, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!36 = !DILocation(line: 15, scope: !35)
!37 = !DILocation(line: 15, scope: !38)
!38 = !DILexicalBlockFile(scope: !35, file: !5, discriminator: 0)
!39 = distinct !DISubprogram(linkageName: "_GLOBAL__sub_I_t.cpp", scope: !1, file: !1, type: !40, isLocal: true, isDefinition: true, flags: DIFlagArtificial, isOptimized: false, unit: !0, variables: !2)
!40 = !DISubroutineType(types: !2)
!41 = !DILocation(line: 0, scope: !39)
