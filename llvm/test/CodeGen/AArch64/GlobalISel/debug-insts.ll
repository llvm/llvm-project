; RUN: llc -global-isel -mtriple=aarch64 %s -stop-after=irtranslator -o - | FileCheck %s
; RUN: llc -mtriple=aarch64 -global-isel --global-isel-abort=0 -o /dev/null

; CHECK-LABEL: name: debug_declare
; CHECK: stack:
; CHECK:    - { id: {{.*}}, name: in.addr, offset: {{.*}}, size: {{.*}}, alignment: {{.*}}, di-variable: '!11',
; CHECK-NEXT:   di-expression: '!12', di-location: '!13' }
; CHECK: DBG_VALUE debug-use %0(s32), debug-use _, !11, !12, debug-location !13
define void @debug_declare(i32 %in) #0 !dbg !7 {
entry:
  %in.addr = alloca i32, align 4
  store i32 %in, i32* %in.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %in.addr, metadata !11, metadata !12), !dbg !13
  call void @llvm.dbg.declare(metadata i32 %in, metadata !11, metadata !12), !dbg !13
  ret void, !dbg !14
}

; CHECK-LABEL: name: debug_declare_vla
; CHECK: DBG_VALUE debug-use %{{[0-9]+}}(p0), debug-use _, !11, !12, debug-location !13
define void @debug_declare_vla(i32 %in) #0 !dbg !7 {
entry:
  %vla.addr = alloca i32, i32 %in
  call void @llvm.dbg.declare(metadata i32* %vla.addr, metadata !11, metadata !12), !dbg !13
  ret void, !dbg !14
}

; CHECK-LABEL: name: debug_value
; CHECK: [[IN:%[0-9]+]](s32) = COPY %w0
define void @debug_value(i32 %in) #0 !dbg !7 {
  %addr = alloca i32
; CHECK: DBG_VALUE debug-use [[IN]](s32), debug-use _, !11, !12, debug-location !13
  call void @llvm.dbg.value(metadata i32 %in, i64 0, metadata !11, metadata !12), !dbg !13
  store i32 %in, i32* %addr
; CHECK: DBG_VALUE debug-use %1(p0), debug-use _, !11, !15, debug-location !13
  call void @llvm.dbg.value(metadata i32* %addr, i64 0, metadata !11, metadata !15), !dbg !13
; CHECK: DBG_VALUE 123, 0, !11, !12, debug-location !13
  call void @llvm.dbg.value(metadata i32 123, i64 0, metadata !11, metadata !12), !dbg !13
; CHECK: DBG_VALUE float 1.000000e+00, 0, !11, !12, debug-location !13
  call void @llvm.dbg.value(metadata float 1.000000e+00, i64 0, metadata !11, metadata !12), !dbg !13
; CHECK: DBG_VALUE _, 0, !11, !12, debug-location !13
  call void @llvm.dbg.value(metadata i32* null, i64 0, metadata !11, metadata !12), !dbg !13
  ret void
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 4.0.0 (trunk 289075) (llvm/trunk 289080)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "tmp.c", directory: "/Users/tim/llvm/build")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"clang version 4.0.0 (trunk 289075) (llvm/trunk 289080)"}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocalVariable(name: "in", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!12 = !DIExpression()
!13 = !DILocation(line: 1, column: 14, scope: !7)
!14 = !DILocation(line: 2, column: 1, scope: !7)
!15 = !DIExpression(DW_OP_deref)
