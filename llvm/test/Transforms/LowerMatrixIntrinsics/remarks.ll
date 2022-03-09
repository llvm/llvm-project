; REQUIRES: aarch64-registered-target

; This test needs to be target specific due to the cost estimate in the output.

; RUN: opt -passes=lower-matrix-intrinsics -pass-remarks=lower-matrix-intrinsics -mtriple=arm64-apple-iphoneos < %s 2>&1 | FileCheck %s

; CHECK-LABEL: remark: test.h:40:20: Lowered with 6 stores, 6 loads, 24 compute ops
; CHECK-NEXT: store(
; CHECK-NEXT:  transpose.2x6.double(load(addr %A)),
; CHECK-NEXT:  addr %B)
define void @transpose(<12 x double>* %A, <12 x double>* %B) !dbg !23 {
  %load = load <12 x double>, <12 x double>* %A, !dbg !24
  %t = call <12 x double> @llvm.matrix.transpose.v12f64.v12f64(<12 x double> %load, i32 2, i32 6), !dbg !24
  store <12 x double> %t, <12 x double>* %B, !dbg !24
  ret void
}

; CHECK-LABEL: remark: test.h:50:20: Lowered with 2 stores, 12 loads, 22 compute ops
; CHECK-NEXT:  store(
; CHECK-NEXT:   multiply.2x6.6x2.double(
; CHECK-NEXT:    load(addr %A),
; CHECK-NEXT:    load(addr %B)),
; CHECK-NEXT:   addr %C)
define void @multiply(<12 x double>* %A, <12 x double>* %B, <4 x double>* %C) !dbg !25 {
  %A.matrix = load <12 x double>, <12 x double>* %A, !dbg !26
  %B.matrix = load <12 x double>, <12 x double>* %B, !dbg !26
  %t = call <4 x double> @llvm.matrix.multiply(<12 x double> %A.matrix, <12 x double> %B.matrix, i32 2, i32 6, i32 2), !dbg !26
  store <4 x double> %t, <4 x double>* %C, !dbg !26
  ret void
}

; CHECK-LABEL: remark: test.h:60:20: Lowered with 6 stores, 6 loads, 0 compute ops
; CHECK-NEXT:  store(
; CHECK-NEXT:   column.major.load.3x3.double(addr %A, 5),
; CHECK-NEXT:   addr %B)
define void @column.major.load(double* %A, <9 x double>* %B) !dbg !27 {
  %A.matrix = call <9 x double> @llvm.matrix.column.major.load(double* %A, i64 5, i1 false, i32 3, i32 3), !dbg !28
  store <9 x double> %A.matrix, <9 x double>* %B, !dbg !28
  ret void
}

; CHECK-LABEL: remark: test.h:70:20: Lowered with 6 stores, 6 loads, 0 compute ops
; CHECK-NEXT:  column.major.store.3x3.double(
; CHECK-NEXT:   column.major.load.3x3.double(addr %A, 5),
; CHECK-NEXT:   addr %B,
; CHECK-NEXT:   10)
define void @column.major.store(double* %A, double* %B) !dbg !29 {
  %A.matrix = call <9 x double> @llvm.matrix.column.major.load(double* %A, i64 5, i1 false, i32 3, i32 3), !dbg !30
  call void @llvm.matrix.column.major.store(<9 x double> %A.matrix, double* %B, i64 10, i1 false, i32 3, i32 3), !dbg !30
  ret void
}

; CHECK-LABEL: remark: test.h:80:20: Lowered with 6 stores, 6 loads, 12 compute ops
; CHECK-NEXT:  column.major.store.3x3.double(
; CHECK-NEXT:   fmul(
; CHECK-NEXT:    fadd(
; CHECK-NEXT:     column.major.load.3x3.double(addr %A, 5)
; CHECK-NEXT:     (reused) column.major.load.3x3.double(addr %A, 5)),
; CHECK-NEXT:    (reused) column.major.load.3x3.double(addr %A, 5)),
; CHECK-NEXT:   addr %B,
; CHECK-NEXT:   10)

define void @binaryops(double* %A, double* %B) !dbg !31 {
  %A.matrix = call <9 x double> @llvm.matrix.column.major.load(double* %A, i64 5, i1 false, i32 3, i32 3), !dbg !32
  %R1.matrix = fadd <9 x double> %A.matrix, %A.matrix, !dbg !32
  %R2.matrix = fmul <9 x double> %R1.matrix, %A.matrix, !dbg !32
  call void @llvm.matrix.column.major.store(<9 x double> %R2.matrix, double* %B, i64 10, i1 false, i32 3, i32 3), !dbg !32
  ret void
}

; CHECK-LABEL: remark: test.h:90:20: Lowered with 6 stores, 6 loads, 12 compute ops
; CHECK-NEXT:  column.major.store.3x3.double(
; CHECK-NEXT:   fmul(
; CHECK-NEXT:    fadd(
; CHECK-NEXT:     column.major.load.3x3.double(addr %A, 5)
; CHECK-NEXT:     (reused) column.major.load.3x3.double(addr %A, 5)),
; CHECK-NEXT:    (reused) column.major.load.3x3.double(addr %A, 5)),
; CHECK-NEXT:   addr %B,
; CHECK-NEXT:   10)
; CHECK-NEXT:  remark: test.h:90:20: Lowered with 2 stores, 12 loads, 22 compute ops
; CHECK-NEXT:  store(
; CHECK-NEXT:   multiply.2x6.6x2.double(
; CHECK-NEXT:    load(addr %C),
; CHECK-NEXT:    load(addr %D)),
; CHECK-NEXT:   addr %E)

define void @multiple_expressions(double* %A, double* %B, <12 x double>* %C, <12 x double>* %D, <4 x double>* %E) !dbg !33 {
  %A.matrix = call <9 x double> @llvm.matrix.column.major.load(double* %A, i64 5, i1 false, i32 3, i32 3), !dbg !34
  %R1.matrix = fadd <9 x double> %A.matrix, %A.matrix, !dbg !34
  %R2.matrix = fmul <9 x double> %R1.matrix, %A.matrix, !dbg !34
  call void @llvm.matrix.column.major.store(<9 x double> %R2.matrix, double* %B, i64 10, i1 false, i32 3, i32 3), !dbg !34

  %C.matrix = load <12 x double>, <12 x double>* %C, !dbg !34
  %D.matrix = load <12 x double>, <12 x double>* %D, !dbg !34
  %Mult.matrix = call <4 x double> @llvm.matrix.multiply(<12 x double> %C.matrix, <12 x double> %D.matrix, i32 2, i32 6, i32 2), !dbg !34
  store <4 x double> %Mult.matrix, <4 x double>* %E, !dbg !34

  ret void
}

; CHECK-LABEL: remark: test.h:100:20: Lowered with 6 stores, 6 loads, 12 compute ops
; CHECK-NEXT:  column.major.store.3x3.double(
; CHECK-NEXT:   fmul(
; CHECK-NEXT:    fadd(
; CHECK-NEXT:     column.major.load.3x3.double(addr %A, 5)
; CHECK-NEXT:     (reused) column.major.load.3x3.double(addr %A, 5)),
; CHECK-NEXT:    (reused) column.major.load.3x3.double(addr %A, 5)),
; CHECK-NEXT:   addr %B,
; CHECK-NEXT:   10)
define void @stackaddresses(double* %A, double* %B) !dbg !35 {
  %A.matrix = call <9 x double> @llvm.matrix.column.major.load(double* %A, i64 5, i1 false, i32 3, i32 3), !dbg !36
  %R1.matrix = fadd <9 x double> %A.matrix, %A.matrix, !dbg !36
  %R2.matrix = fmul <9 x double> %R1.matrix, %A.matrix, !dbg !36
  call void @llvm.matrix.column.major.store(<9 x double> %R2.matrix, double* %B, i64 10, i1 false, i32 3, i32 3), !dbg !36
  ret void
}

; CHECK-LABEL: remark: test.h:30:20: Lowered with 10 stores, 9 loads, 30 compute ops
; CHECK-NEXT:  store(
; CHECK-NEXT:   transpose.5x3.double(load(addr %A)),
; CHECK-NEXT:   stack addr %s1)
%S1 = type {<15 x double>*}
define void @get_underlying_object(%S1* %A) !dbg !21 {
entry:
  %s1 = alloca <15 x double>, !dbg !22
  %a1 = getelementptr %S1, %S1* %A, i32 0, i32 0, !dbg !22
  %a2 = load <15 x double>*, <15 x double>** %a1, !dbg !22
  %av = load <15 x double>, <15 x double>* %a2, !dbg !22

  %s2 = bitcast <15 x double>* %s1 to i64*, !dbg !22
  %s3 = bitcast i64* %s2 to <15 x double>*, !dbg !22

  %t = call <15 x double> @llvm.matrix.transpose.v15f64.v15f64(<15 x double> %av, i32 5, i32 3), !dbg !22

  store <15 x double> %t, <15 x double>* %s3, !dbg !22
  ret void
}

declare <12 x double> @llvm.matrix.transpose.v12f64.v12f64(<12 x double>, i32, i32)
declare <4 x double> @llvm.matrix.multiply(<12 x double>, <12 x double>, i32, i32, i32)
declare <9 x double> @llvm.matrix.column.major.load(double*, i64, i1, i32, i32)
declare <15 x double> @llvm.matrix.transpose.v15f64.v15f64(<15 x double>, i32, i32)
declare void @llvm.matrix.column.major.store(<9 x double>, double*, i64, i1, i32, i32)


!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.h", directory: "/test")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}

!6 = !DISubroutineType(types: !7)
!7 = !{null, !8, !8, !11}
!8 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !9)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 32, align: 32)
!10 = !DIBasicType(name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!11 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!12 = !{!13}
!13 = !DILocalVariable(name: "a", arg: 1, scope: !5, file: !1, line: 1, type: !8)
!14 = !DILocation(line: 1, column: 27, scope: !5)

!5 = distinct !DISubprogram(name: "fn1", scope: !1, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)
!19 = !DILocation(line: 10, column: 20, scope: !5)
!20 = !DILocation(line: 10, column: 10, scope: !5)

!21 = distinct !DISubprogram(name: "fn2", scope: !1, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)
!22 = !DILocation(line: 30, column: 20, scope: !21)

!23 = distinct !DISubprogram(name: "fn3", scope: !1, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)
!24 = !DILocation(line: 40, column: 20, scope: !23)

!25 = distinct !DISubprogram(name: "fn4", scope: !1, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)
!26 = !DILocation(line: 50, column: 20, scope: !25)

!27 = distinct !DISubprogram(name: "fn5", scope: !1, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)
!28 = !DILocation(line: 60, column: 20, scope: !27)

!29 = distinct !DISubprogram(name: "fn6", scope: !1, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)
!30 = !DILocation(line: 70, column: 20, scope: !29)

!31 = distinct !DISubprogram(name: "fn7", scope: !1, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)
!32 = !DILocation(line: 80, column: 20, scope: !31)

!33 = distinct !DISubprogram(name: "fn8", scope: !1, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)
!34 = !DILocation(line: 90, column: 20, scope: !33)

!35 = distinct !DISubprogram(name: "fn9", scope: !1, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)
!36 = !DILocation(line: 100, column: 20, scope: !35)
