; RUN: opt < %s -passes=instcombine -instcombine-lower-dbg-declare=0 -S \
; RUN:      | FileCheck %s --check-prefix=CHECK --check-prefix=NOLOWER
; RUN: opt < %s -passes=instcombine -instcombine-lower-dbg-declare=1 -S | FileCheck %s

; RUN: opt < %s -passes=instcombine -instcombine-lower-dbg-declare=0 -S --try-experimental-debuginfo-iterators \
; RUN:      | FileCheck %s --check-prefix=CHECK --check-prefix=NOLOWER
; RUN: opt < %s -passes=instcombine -instcombine-lower-dbg-declare=1 -S --try-experimental-debuginfo-iterators | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64--linux"

%struct.TwoRegs = type { i64, i64 }

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare i64 @llvm.objectsize.i64.p0(ptr, i1) nounwind readnone

declare ptr @passthru_callee(ptr, i32, i64, i64)

define ptr @passthru(ptr %a, i32 %b, i64 %c) !dbg !1 {
entry:
  %a.addr = alloca ptr, align 8
  %b.addr = alloca i32, align 4
  %c.addr = alloca i64, align 8
  store ptr %a, ptr %a.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %a.addr, metadata !0, metadata !DIExpression()), !dbg !16
  store i32 %b, ptr %b.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %b.addr, metadata !7, metadata !DIExpression()), !dbg !18
  store i64 %c, ptr %c.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %c.addr, metadata !9, metadata !DIExpression()), !dbg !20
  %tmp = load ptr, ptr %a.addr, align 8, !dbg !21
  %tmp1 = load i32, ptr %b.addr, align 4, !dbg !21
  %tmp2 = load i64, ptr %c.addr, align 8, !dbg !21
  %tmp3 = load ptr, ptr %a.addr, align 8, !dbg !21
  %0 = call i64 @llvm.objectsize.i64.p0(ptr %tmp3, i1 false), !dbg !21
  %call = call ptr @passthru_callee(ptr %tmp, i32 %tmp1, i64 %tmp2, i64 %0), !dbg !21
  ret ptr %call, !dbg !21
}

; CHECK-LABEL: define ptr @passthru(ptr %a, i32 %b, i64 %c)
; CHECK-NOT: alloca
; CHECK-NOT: store
; CHECK-NOT: #dbg_declare
; CHECK: #dbg_value(ptr %a, {{.*}})
; CHECK-NOT: store
; CHECK: #dbg_value(i32 %b, {{.*}})
; CHECK-NOT: store
; CHECK: #dbg_value(i64 %c, {{.*}})
; CHECK-NOT: store
; CHECK: call ptr @passthru_callee(ptr %a, i32 %b, i64 %c, i64 %{{.*}})

declare void @tworegs_callee(i64, i64)

; Lowering dbg.declare in instcombine doesn't handle this case very well.

define void @tworegs(i64 %o.coerce0, i64 %o.coerce1) !dbg !31 {
entry:
  %o = alloca %struct.TwoRegs, align 8
  %0 = getelementptr inbounds { i64, i64 }, ptr %o, i32 0, i32 0
  store i64 %o.coerce0, ptr %0, align 8
  %1 = getelementptr inbounds { i64, i64 }, ptr %o, i32 0, i32 1
  store i64 %o.coerce1, ptr %1, align 8
  call void @llvm.dbg.declare(metadata ptr %o, metadata !35, metadata !DIExpression()), !dbg !32
  %2 = getelementptr inbounds { i64, i64 }, ptr %o, i32 0, i32 0, !dbg !33
  %3 = load i64, ptr %2, align 8, !dbg !33
  %4 = getelementptr inbounds { i64, i64 }, ptr %o, i32 0, i32 1, !dbg !33
  %5 = load i64, ptr %4, align 8, !dbg !33
  call void @tworegs_callee(i64 %3, i64 %5), !dbg !33
  ret void, !dbg !33
}

; NOLOWER-LABEL: define void @tworegs(i64 %o.coerce0, i64 %o.coerce1)
; NOLOWER-NOT: alloca
; NOLOWER-NOT: store
; NOLOWER-NOT: #dbg_declare
; Here we want to find:  call void @llvm.dbg.value(metadata i64 %o.coerce0, metadata [[VARIABLE_O]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 64))
; NOLOWER: #dbg_value(i64 undef, {{.*}})
; NOLOWER-NOT: store
; Here we want to find:  call void @llvm.dbg.value(metadata i64 %o.coerce1, metadata [[VARIABLE_O]], metadata !DIExpression(DW_OP_LLVM_fragment, 64, 64))
; NOLOWER: #dbg_value(i64 undef, {{.*}})
; NOLOWER-NOT: store
; NOLOWER: call void @tworegs_callee(i64 %o.coerce0, i64 %o.coerce1)


!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!30}

!0 = !DILocalVariable(name: "a", line: 78, arg: 1, scope: !1, file: !2, type: !6)
!1 = distinct !DISubprogram(name: "passthru", line: 79, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !3, scopeLine: 79, file: !27, scope: !2, type: !4, retainedNodes: !25)
!2 = !DIFile(filename: "string.h", directory: "Game")
!3 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.0 (trunk 127710)", isOptimized: true, emissionKind: FullDebug, file: !28, enums: !29, retainedTypes: !29)
!4 = !DISubroutineType(types: !5)
!5 = !{!6}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, scope: !3, baseType: null)
!7 = !DILocalVariable(name: "b", line: 78, arg: 2, scope: !1, file: !2, type: !8)
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DILocalVariable(name: "c", line: 78, arg: 3, scope: !1, file: !2, type: !12)
!12 = !DIBasicType(tag: DW_TAG_base_type, name: "long unsigned int", size: 64, align: 64, encoding: DW_ATE_unsigned)
!16 = !DILocation(line: 78, column: 28, scope: !1)
!18 = !DILocation(line: 78, column: 40, scope: !1)
!20 = !DILocation(line: 78, column: 54, scope: !1)
!21 = !DILocation(line: 80, column: 3, scope: !22)
!22 = distinct !DILexicalBlock(line: 80, column: 3, file: !27, scope: !23)
!23 = distinct !DILexicalBlock(line: 79, column: 1, file: !27, scope: !1)
!25 = !{!0, !7, !9}
!27 = !DIFile(filename: "string.h", directory: "Game")
!28 = !DIFile(filename: "bits.c", directory: "Game")
!29 = !{}
!30 = !{i32 1, !"Debug Info Version", i32 3}

!31 = distinct !DISubprogram(name: "tworegs", scope: !28, file: !28, line: 4, type: !4, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: true, unit: !3, retainedNodes: !34)
!32 = !DILocation(line: 4, column: 23, scope: !31)
!33 = !DILocation(line: 5, column: 3, scope: !31)
!34 = !{!35}
!35 = !DILocalVariable(name: "o", arg: 1, scope: !31, file: !28, line: 4, type: !36)
!36 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "TwoRegs", file: !28, line: 1, size: 128, elements: !37)
!37 = !{!38, !39}
!38 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !36, file: !28, line: 1, baseType: !12, size: 64)
!39 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !36, file: !28, line: 1, baseType: !12, size: 64)
!40 = !DISubroutineType(types: !41)
!41 = !{!36}
