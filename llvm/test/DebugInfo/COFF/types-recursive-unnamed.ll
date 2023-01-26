; RUN: not --crash llc < %s -filetype=obj 2>&1 | FileCheck %s
;
; Verify the compiler produces an error message when trying to emit circular
; references to unnamed structs which are not supported in CodeView debug
; information.
; 
; -- types-recursive-unnamed.cpp begin -----------------------------------------
; struct named_struct {
;   struct {
;     void method() {}
;   } unnamed_struct_with_method;
;   void anchor();
; };
; void named_struct::anchor() {}
; -- types-recursive-unnamed.cpp end -------------------------------------------
;
; To rebuild the reproducer:
;   1. First, compile the source code:
;      $ clang -S -emit-llvm -g -gcodeview unnamed.cpp
;   2. Remove all "name" and "identifier" attributes with a value matching
;      the form: "<unnamed-type-".
;
; CHECK: LLVM ERROR: cannot debug circular reference to unnamed type

; ModuleID = 'types-recursive-unnamed.cpp'
source_filename = "types-recursive-unnamed.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24210"

%struct.named_struct = type { %struct.anon }
%struct.anon = type { i8 }

; Function Attrs: noinline nounwind uwtable
define void @"\01?anchor@named_struct@@QEAAXXZ"(ptr %this) #0 align 2 !dbg !7 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %this.addr, metadata !21, metadata !23), !dbg !24
  %this1 = load ptr, ptr %this.addr, align 8
  ret void, !dbg !25
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 5.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "types-recursive-unnamed.cpp", directory: "C:\5Cpath\5Cto\5Cdirectory", checksumkind: CSK_MD5, checksum: "59a90813f3338cfe690d9664215089df")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"clang version 5.0.0"}
!7 = distinct !DISubprogram(name: "anchor", linkageName: "\01?anchor@named_struct@@QEAAXXZ", scope: !8, file: !1, line: 7, type: !18, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !17, retainedNodes: !2)
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "named_struct", file: !1, line: 1, size: 8, elements: !9, identifier: ".?AUnamed_struct@@")
!9 = !{!10, !16, !17}
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, scope: !8, file: !1, line: 2, size: 8, elements: !11)
!11 = !{!12}
!12 = !DISubprogram(name: "method", linkageName: "\01?method@<unnamed-type-unnamed_struct_with_method>@named_struct@@QEAAXXZ", scope: !10, file: !1, line: 3, type: !13, isLocal: false, isDefinition: false, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false)
!13 = !DISubroutineType(types: !14)
!14 = !{null, !15}
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!16 = !DIDerivedType(tag: DW_TAG_member, name: "unnamed_struct_with_method", scope: !8, file: !1, line: 4, baseType: !10, size: 8)
!17 = !DISubprogram(name: "anchor", linkageName: "\01?anchor@named_struct@@QEAAXXZ", scope: !8, file: !1, line: 5, type: !18, isLocal: false, isDefinition: false, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: false)
!18 = !DISubroutineType(types: !19)
!19 = !{null, !20}
!20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!21 = !DILocalVariable(name: "this", arg: 1, scope: !7, type: !22, flags: DIFlagArtificial | DIFlagObjectPointer)
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!23 = !DIExpression()
!24 = !DILocation(line: 0, scope: !7)
!25 = !DILocation(line: 7, column: 30, scope: !7)
