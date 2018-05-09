; RUN: llc %s -stop-after=livedebugvars -o - | FileCheck %s
source_filename = "/tmp/t.ll"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.13"

%TSb = type <{ i1 }>

declare swiftcc i1 @f()

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

; Function Attrs: nounwind readnone speculatable
define swiftcc void @g() #0 !dbg !5 {
entry:
  %0 = alloca %TSb, align 1
  %1 = call swiftcc i1 @f(), !dbg !7
  ; CHECK: DBG_VALUE debug-use $rcx, debug-use $noreg, !8, !DIExpression(), debug-location !7
  call void @llvm.dbg.value(metadata i1 %1, metadata !8, metadata !DIExpression()), !dbg !7
  %2 = getelementptr inbounds %TSb, %TSb* %0, i32 0, i32 0, !dbg !7
  store i1 %1, i1* %2, align 1, !dbg !7
  %3 = zext i1 %1 to i64, !dbg !7
  call void asm sideeffect "", "r"(i64 %3), !dbg !7
  ret void, !dbg !7
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !1, isOptimized: false, runtimeVersion: 4, emissionKind: FullDebug, enums: !2, imports: !2)
!1 = !DIFile(filename: "t.swift", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "g", scope: !0, file: !1, line: 2, type: !6, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: false, unit: !0, retainedNodes: !2)
!6 = !DISubroutineType(types: !2)
!7 = !DILocation(line: 4, scope: !5)
!8 = !DILocalVariable(name: "hasInput", scope: !5, file: !1, line: 3, type: !9)
!9 = !DICompositeType(tag: DW_TAG_structure_type, name: "Bool", scope: !11, file: !10, size: 8, elements: !2, runtimeLang: DW_LANG_Swift, identifier: "_T0SbD")
!10 = !DIFile(filename: "Swift.swiftmodule", directory: "/usr/lib/swift/macosx/x86_64")
!11 = !DIModule(scope: null, name: "Swift", includePath: "/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk")
