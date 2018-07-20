; RUN: llc < %s -filetype=obj | llvm-readobj - -codeview | FileCheck %s
;
; Verify CodeView emission does not emit forward references for unnamed
; structs/unions.  If a forward reference is emitted for an unnamed composite
; type then Visual Studio will not be able to display the value.
;
;   Previous values displayed by Visual Studio:
;      local.unnamed_struct    {...}
;      local.unnamed_union     {...}
;
;   New values displayed by Visual Studio:
;      local.unnamed_struct    {m3=66 'B' }
;      local.unnamed_union     {m1=65 m2=65 'A' }
;
; The reproducer:
;   $ cat unnamed.c
;   struct named_struct {
;     int    id;
;     union {
;       int  m1;
;       char m2;
;     } unnamed_union;
;     struct {
;       char m3;
;     } unnamed_struct;
;   };
;   
;   int main()
;   {
;     struct named_struct local;
;   
;     local.id = 1;
;     local.unnamed_union.m1 = 65;
;     local.unnamed_struct.m3 = 'B';
;   
;     return 0;
;   }
;
; To regenerate the IR below:
;   $ clang unnamed.c -S -emit-llvm -g -gcodeview
;
; CHECK:      FieldList ([[UnnamedUnionFieldList:.*]]) {
; CHECK-NEXT:   TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK-NEXT:   DataMember {
; CHECK-NEXT:     TypeLeafKind: LF_MEMBER (0x150D)
; CHECK-NEXT:     AccessSpecifier: Public (0x3)
; CHECK-NEXT:     Type: int (0x74)
; CHECK-NEXT:     FieldOffset: 0x0
; CHECK-NEXT:     Name: m1
; CHECK-NEXT:   }
; CHECK-NEXT:   DataMember {
; CHECK-NEXT:     TypeLeafKind: LF_MEMBER (0x150D)
; CHECK-NEXT:     AccessSpecifier: Public (0x3)
; CHECK-NEXT:     Type: char (0x70)
; CHECK-NEXT:     FieldOffset: 0x0
; CHECK-NEXT:     Name: m2
; CHECK-NEXT:   }
; CHECK-NEXT: }
; CHECK:      Union ([[UnnamedUnion:.*]]) {
; CHECK-NEXT:   TypeLeafKind: LF_UNION (0x1506)
; CHECK-NEXT:   MemberCount: 2
; CHECK-NEXT:   Properties [ (0x408)
; CHECK-NEXT:     Nested (0x8)
; CHECK-NEXT:     Sealed (0x400)
; CHECK-NEXT:   ]
; CHECK-NEXT:   FieldList: <field list> ([[UnnamedUnionFieldList]])
; CHECK-NEXT:   SizeOf: 4
; CHECK-NEXT:   Name: named_struct::<unnamed-tag>
; CHECK-NEXT: }
; CHECK:      FieldList ([[UnnamedStructFieldList:.*]]) {
; CHECK-NEXT:   TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK-NEXT:   DataMember {
; CHECK-NEXT:     TypeLeafKind: LF_MEMBER (0x150D)
; CHECK-NEXT:     AccessSpecifier: Public (0x3)
; CHECK-NEXT:     Type: char (0x70)
; CHECK-NEXT:     FieldOffset: 0x0
; CHECK-NEXT:     Name: m3
; CHECK-NEXT:   }
; CHECK-NEXT: }
; CHECK:      Struct ([[UnnamedStruct:.*]]) {
; CHECK-NEXT:   TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK-NEXT:   MemberCount: 1
; CHECK-NEXT:   Properties [ (0x8)
; CHECK-NEXT:     Nested (0x8)
; CHECK-NEXT:   ]
; CHECK-NEXT:   FieldList: <field list> ([[UnnamedStructFieldList]])
; CHECK-NEXT:   DerivedFrom: 0x0
; CHECK-NEXT:   VShape: 0x0
; CHECK-NEXT:   SizeOf: 1
; CHECK-NEXT:   Name: named_struct::<unnamed-tag>
; CHECK-NEXT: }
; CHECK:      FieldList ([[NamedStructFieldList:.*]]) {
; CHECK-NEXT:   TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK-NEXT:   DataMember {
; CHECK-NEXT:     TypeLeafKind: LF_MEMBER (0x150D)
; CHECK-NEXT:     AccessSpecifier: Public (0x3)
; CHECK-NEXT:     Type: int (0x74)
; CHECK-NEXT:     FieldOffset: 0x0
; CHECK-NEXT:     Name: id
; CHECK-NEXT:   }
; CHECK-NEXT:   DataMember {
; CHECK-NEXT:     TypeLeafKind: LF_MEMBER (0x150D)
; CHECK-NEXT:     AccessSpecifier: Public (0x3)
; CHECK-NEXT:     Type: named_struct::<unnamed-tag> ([[UnnamedUnion]])
; CHECK-NEXT:     FieldOffset: 0x4
; CHECK-NEXT:     Name: unnamed_union
; CHECK-NEXT:   }
; CHECK-NEXT:   DataMember {
; CHECK-NEXT:     TypeLeafKind: LF_MEMBER (0x150D)
; CHECK-NEXT:     AccessSpecifier: Public (0x3)
; CHECK-NEXT:     Type: named_struct::<unnamed-tag> ([[UnnamedStruct]])
; CHECK-NEXT:     FieldOffset: 0x8
; CHECK-NEXT:     Name: unnamed_struct
; CHECK-NEXT:   }
; CHECK-NEXT: }
; CHECK:      Struct ({{.*}}) {
; CHECK-NEXT:   TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK-NEXT:   MemberCount: 3
; CHECK-NEXT:   Properties [ (0x0)
; CHECK-NEXT:   ]
; CHECK-NEXT:   FieldList: <field list> ([[NamedStructFieldList]])
; CHECK-NEXT:   DerivedFrom: 0x0
; CHECK-NEXT:   VShape: 0x0
; CHECK-NEXT:   SizeOf: 12
; CHECK-NEXT:   Name: named_struct
; CHECK-NEXT: }

; ModuleID = 'unnamed.c'
source_filename = "unnamed.c"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24210"

%struct.named_struct = type { i32, %union.anon, %struct.anon }
%union.anon = type { i32 }
%struct.anon = type { i8 }

; Function Attrs: noinline nounwind uwtable
define i32 @main() #0 !dbg !7 {
entry:
  %retval = alloca i32, align 4
  %local = alloca %struct.named_struct, align 4
  store i32 0, i32* %retval, align 4
  call void @llvm.dbg.declare(metadata %struct.named_struct* %local, metadata !11, metadata !25), !dbg !26
  %id = getelementptr inbounds %struct.named_struct, %struct.named_struct* %local, i32 0, i32 0, !dbg !27
  store i32 1, i32* %id, align 4, !dbg !28
  %unnamed_union = getelementptr inbounds %struct.named_struct, %struct.named_struct* %local, i32 0, i32 1, !dbg !29
  %m1 = bitcast %union.anon* %unnamed_union to i32*, !dbg !30
  store i32 65, i32* %m1, align 4, !dbg !31
  %unnamed_struct = getelementptr inbounds %struct.named_struct, %struct.named_struct* %local, i32 0, i32 2, !dbg !32
  %m3 = getelementptr inbounds %struct.anon, %struct.anon* %unnamed_struct, i32 0, i32 0, !dbg !33
  store i8 66, i8* %m3, align 4, !dbg !34
  ret i32 0, !dbg !35
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 5.0.0 (trunk)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "unnamed.c", directory: "C:\5Cpath\5Cto\5Cdirectory", checksumkind: CSK_MD5, checksum: "a1874da39665a126d6949d929fbd4818")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"clang version 5.0.0 (trunk)"}
!7 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 12, type: !8, isLocal: false, isDefinition: true, scopeLine: 13, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocalVariable(name: "local", scope: !7, file: !1, line: 14, type: !12)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "named_struct", file: !1, line: 1, size: 96, elements: !13)
!13 = !{!14, !15, !21}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "id", scope: !12, file: !1, line: 2, baseType: !10, size: 32)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "unnamed_union", scope: !12, file: !1, line: 6, baseType: !16, size: 32, offset: 32)
!16 = distinct !DICompositeType(tag: DW_TAG_union_type, scope: !12, file: !1, line: 3, size: 32, elements: !17)
!17 = !{!18, !19}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "m1", scope: !16, file: !1, line: 4, baseType: !10, size: 32)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "m2", scope: !16, file: !1, line: 5, baseType: !20, size: 8)
!20 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "unnamed_struct", scope: !12, file: !1, line: 9, baseType: !22, size: 8, offset: 64)
!22 = distinct !DICompositeType(tag: DW_TAG_structure_type, scope: !12, file: !1, line: 7, size: 8, elements: !23)
!23 = !{!24}
!24 = !DIDerivedType(tag: DW_TAG_member, name: "m3", scope: !22, file: !1, line: 8, baseType: !20, size: 8)
!25 = !DIExpression()
!26 = !DILocation(line: 14, column: 23, scope: !7)
!27 = !DILocation(line: 16, column: 9, scope: !7)
!28 = !DILocation(line: 16, column: 12, scope: !7)
!29 = !DILocation(line: 17, column: 9, scope: !7)
!30 = !DILocation(line: 17, column: 23, scope: !7)
!31 = !DILocation(line: 17, column: 26, scope: !7)
!32 = !DILocation(line: 18, column: 9, scope: !7)
!33 = !DILocation(line: 18, column: 24, scope: !7)
!34 = !DILocation(line: 18, column: 27, scope: !7)
!35 = !DILocation(line: 20, column: 3, scope: !7)
