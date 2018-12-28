; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv -spirv-mem2reg=false
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o %t.ll

; RUN: llc -mtriple=%triple -O0 %t.ll -o /dev/null

define void @CGRectStandardize(i32* sret %agg.result, i32* byval %rect) nounwind ssp !dbg !0 {
entry:
  call void @llvm.dbg.declare(metadata i32* %rect, metadata !23, metadata !DIExpression()), !dbg !24
  ret void
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind


!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!27}
!0 = distinct !DISubprogram(name: "CGRectStandardize", linkageName: "CGRectStandardize", line: 54, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !2, file: !1, scope: null, type: !28)
!1 = !DIFile(filename: "GSFusedSilica.m", directory: "/Volumes/Data/Users/sabre/Desktop")
!2 = distinct !DICompileUnit(language: DW_LANG_ObjC, producer: "clang version 2.9 (trunk 115292)", isOptimized: true, runtimeVersion: 1, emissionKind: FullDebug, file: !25, enums: !26, retainedTypes: !26)
!5 = !DIDerivedType(tag: DW_TAG_typedef, name: "CGRect", line: 49, file: !25, baseType: null)
!23 = !DILocalVariable(name: "rect", line: 53, arg: 2, scope: !0, file: !1, type: !5)
!24 = !DILocation(line: 53, column: 33, scope: !0)
!25 = !DIFile(filename: "GSFusedSilica.m", directory: "/Volumes/Data/Users/sabre/Desktop")
!26 = !{}
!27 = !{i32 1, !"Debug Info Version", i32 3}
!28 = !DISubroutineType(types: !29)
!29 = !{null}
target triple = "spir64-unknown-unknown"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
