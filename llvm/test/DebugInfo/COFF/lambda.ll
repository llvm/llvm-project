; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s
;
; Verify lambda routines are emitted properly in CodeView.
;
; The original source code:
; -----------------------------------------------------------------------------
; int main(int argc, char* argv[], char* arge[]) {
;   auto Lambda = [argc](int count) -> int { return argc == count ? 1 : 0; };
;   return Lambda(0);
; }
; -----------------------------------------------------------------------------
;
; To regenerate the IR below compile the source code:
;   $ clang lambda.cxx -S -emit-llvm -g -gcodeview
;
; CHECK:      FieldList ([[FIELDLIST_ID:0x[0-9A-F]+]]) {
; CHECK-NEXT:   TypeLeafKind: LF_FIELDLIST ({{.*}})
; CHECK-NEXT:   DataMember {
; CHECK-NEXT:     TypeLeafKind: LF_MEMBER ({{.*}})
; CHECK-NEXT:     AccessSpecifier: Private ({{.*}})
; CHECK-NEXT:     Type: int ({{.*}})
; CHECK-NEXT:     FieldOffset: {{.*}}
; CHECK-NEXT:     Name: argc
; CHECK-NEXT:   }
; CHECK-NEXT:   OneMethod {
; CHECK-NEXT:     TypeLeafKind: LF_ONEMETHOD (0x1511)
; CHECK-NEXT:     AccessSpecifier: Public (0x3)
; CHECK-NEXT:     Type: int main::<unnamed-tag>::(int) ({{.*}})
; CHECK-NEXT:     Name: operator()
; CHECK-NEXT:   }
; CHECK-NEXT: }
; CHECK-NEXT: Class ([[CLASS_ID:0x[0-9A-F]+]]) {
; CHECK-NEXT:   TypeLeafKind: LF_CLASS ({{.*}})
; CHECK-NEXT:   MemberCount: {{.*}}
; CHECK-NEXT:   Properties [ ({{.*}})
; CHECK-NEXT:     HasUniqueName ({{.*}})
; CHECK-NEXT:     Scoped ({{.*}})
; CHECK-NEXT:   ]
; CHECK-NEXT:   FieldList: <field list> ([[FIELDLIST_ID]])
; CHECK-NEXT:   DerivedFrom: {{.*}}
; CHECK-NEXT:   VShape: {{.*}}
; CHECK-NEXT:   SizeOf: {{.*}}
; CHECK-NEXT:   Name: main::<unnamed-tag>
; CHECK-NEXT:   LinkageName: {{.*lambda.*}}
; CHECK-NEXT: }
;             LocalSym {
;               Kind: S_LOCAL ({{.*}})
; CHECK:        Type: main::<unnamed-tag> ([[CLASS_ID]])
;               Flags [ (0x0)
;               ]
; CHECK:        VarName: Lambda
;             }

; ModuleID = 'lambda.cxx'
source_filename = "lambda.cxx"
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc19.0.24210"

%class.anon = type { i32 }

; Function Attrs: noinline norecurse optnone
define dso_local i32 @main(i32 %argc, i8** %argv, i8** %arge) #0 !dbg !8 {
entry:
  %retval = alloca i32, align 4
  %arge.addr = alloca i8**, align 4
  %argv.addr = alloca i8**, align 4
  %argc.addr = alloca i32, align 4
  %Lambda = alloca %class.anon, align 4
  store i32 0, i32* %retval, align 4
  store i8** %arge, i8*** %arge.addr, align 4
  call void @llvm.dbg.declare(metadata i8*** %arge.addr, metadata !15, metadata !DIExpression()), !dbg !16
  store i8** %argv, i8*** %argv.addr, align 4
  call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !17, metadata !DIExpression()), !dbg !16
  store i32 %argc, i32* %argc.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !18, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata %class.anon* %Lambda, metadata !19, metadata !DIExpression()), !dbg !28
  %0 = getelementptr inbounds %class.anon, %class.anon* %Lambda, i32 0, i32 0, !dbg !28
  %1 = load i32, i32* %argc.addr, align 4, !dbg !28
  store i32 %1, i32* %0, align 4, !dbg !28
  %call = call x86_thiscallcc i32 @"??R<lambda_0>@?0??main@@9@QBE@H@Z"(%class.anon* %Lambda, i32 0), !dbg !29
  ret i32 %call, !dbg !29
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind optnone
define internal x86_thiscallcc i32 @"??R<lambda_0>@?0??main@@9@QBE@H@Z"(%class.anon* %this, i32 %count) #2 align 2 !dbg !30 {
entry:
  %count.addr = alloca i32, align 4
  %this.addr = alloca %class.anon*, align 4
  store i32 %count, i32* %count.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %count.addr, metadata !31, metadata !DIExpression()), !dbg !32
  store %class.anon* %this, %class.anon** %this.addr, align 4
  call void @llvm.dbg.declare(metadata %class.anon** %this.addr, metadata !33, metadata !DIExpression()), !dbg !35
  %this1 = load %class.anon*, %class.anon** %this.addr, align 4
  %0 = getelementptr inbounds %class.anon, %class.anon* %this1, i32 0, i32 0, !dbg !32
  %1 = load i32, i32* %0, align 4, !dbg !32
  %2 = load i32, i32* %count.addr, align 4, !dbg !32
  %cmp = icmp eq i32 %1, %2, !dbg !32
  %3 = zext i1 %cmp to i64, !dbg !32
  %cond = select i1 %cmp, i32 1, i32 0, !dbg !32
  ret i32 %cond, !dbg !32
}

attributes #0 = { noinline norecurse optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 7.0.0 (trunk)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "lambda.cxx", directory: "C:\5Cpath\5Cto", checksumkind: CSK_MD5, checksum: "8d860c432e3763effaf1658460a496c0")
!2 = !{}
!3 = !{i32 1, !"NumRegisterParameters", i32 0}
!4 = !{i32 2, !"CodeView", i32 1}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 1, !"wchar_size", i32 2}
!7 = !{!"clang version 7.0.0 (trunk)"}
!8 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !9, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11, !12, !12}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 32)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 32)
!14 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!15 = !DILocalVariable(name: "arge", arg: 3, scope: !8, file: !1, line: 1, type: !12)
!16 = !DILocation(line: 1, scope: !8)
!17 = !DILocalVariable(name: "argv", arg: 2, scope: !8, file: !1, line: 1, type: !12)
!18 = !DILocalVariable(name: "argc", arg: 1, scope: !8, file: !1, line: 1, type: !11)
!19 = !DILocalVariable(name: "Lambda", scope: !8, file: !1, line: 2, type: !20)
!20 = distinct !DICompositeType(tag: DW_TAG_class_type, scope: !8, file: !1, line: 2, size: 32, flags: DIFlagTypePassByValue, elements: !21, identifier: "??R<lambda_0>@?0??main@@9@QBE@H@Z")
!21 = !{!22, !23}
!22 = !DIDerivedType(tag: DW_TAG_member, name: "argc", scope: !20, file: !1, line: 2, baseType: !11, size: 32)
!23 = !DISubprogram(name: "operator()", scope: !20, file: !1, line: 2, type: !24, isLocal: false, isDefinition: false, scopeLine: 2, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: false)
!24 = !DISubroutineType(cc: DW_CC_BORLAND_thiscall, types: !25)
!25 = !{!11, !26, !11}
!26 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !27, size: 32, flags: DIFlagArtificial | DIFlagObjectPointer)
!27 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !20)
!28 = !DILocation(line: 2, scope: !8)
!29 = !DILocation(line: 3, scope: !8)
!30 = distinct !DISubprogram(name: "operator()", linkageName: "??R<lambda_0>@?0??main@@9@QBE@H@Z", scope: !20, file: !1, line: 2, type: !24, isLocal: true, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !23, retainedNodes: !2)
!31 = !DILocalVariable(name: "count", arg: 2, scope: !30, file: !1, line: 2, type: !11)
!32 = !DILocation(line: 2, scope: !30)
!33 = !DILocalVariable(name: "this", arg: 1, scope: !30, type: !34, flags: DIFlagArtificial | DIFlagObjectPointer)
!34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !27, size: 32)
!35 = !DILocation(line: 0, scope: !30)
