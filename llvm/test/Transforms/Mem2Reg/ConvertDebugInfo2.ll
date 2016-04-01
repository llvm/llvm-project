; RUN: opt -S -mem2reg <%s | FileCheck %s

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare void @foo(i32, i64, i8*)

define void @baz(i32 %a) nounwind ssp !dbg !1 {
; CHECK-LABEL:  entry:
; CHECK-NEXT:     %"alloca point" = bitcast i32 0 to i32{{$}}
; CHECK-NEXT:     call void @llvm.dbg.value(metadata i32 %a,{{.*}}, !dbg
; CHECK-NEXT:     call void @llvm.dbg.value(metadata i32 %a,{{.*}}, !dbg
; CHECK-NEXT:     call void @llvm.dbg.value(metadata i64 55,{{.*}}, !dbg
; CHECK-NEXT:     call void @llvm.dbg.value(metadata i8* bitcast (void (i32)* @baz to i8*),{{.*}}, !dbg
; CHECK-NEXT:     call void @foo({{.*}}, !dbg
; CHECK-NEXT:     br label %return, !dbg
entry:
  %x_addr.i = alloca i32                          ; <i32*> [#uses=2]
  %y_addr.i = alloca i64                          ; <i64*> [#uses=2]
  %z_addr.i = alloca i8*                          ; <i8**> [#uses=2]
  %a_addr = alloca i32                            ; <i32*> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata i32* %a_addr, metadata !0, metadata !DIExpression()), !dbg !7
  store i32 %a, i32* %a_addr
  %0 = load i32, i32* %a_addr, align 4, !dbg !8        ; <i32> [#uses=1]
  call void @llvm.dbg.declare(metadata i32* %x_addr.i, metadata !9, metadata !DIExpression()) nounwind, !dbg !15
  store i32 %0, i32* %x_addr.i
  call void @llvm.dbg.declare(metadata i64* %y_addr.i, metadata !16, metadata !DIExpression()) nounwind, !dbg !15
  store i64 55, i64* %y_addr.i
  call void @llvm.dbg.declare(metadata i8** %z_addr.i, metadata !17, metadata !DIExpression()) nounwind, !dbg !15
  store i8* bitcast (void (i32)* @baz to i8*), i8** %z_addr.i
  %1 = load i32, i32* %x_addr.i, align 4, !dbg !18     ; <i32> [#uses=1]
  %2 = load i64, i64* %y_addr.i, align 8, !dbg !18     ; <i64> [#uses=1]
  %3 = load i8*, i8** %z_addr.i, align 8, !dbg !18     ; <i8*> [#uses=1]
  call void @foo(i32 %1, i64 %2, i8* %3) nounwind, !dbg !18
  br label %return, !dbg !19

; CHECK-LABEL:  return:
; CHECK-NEXT:     ret void, !dbg
return:                                           ; preds = %entry
  ret void, !dbg !19
}

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!22}
!0 = !DILocalVariable(name: "a", line: 8, arg: 1, scope: !1, file: !2, type: !6)
!1 = distinct !DISubprogram(name: "baz", linkageName: "baz", line: 8, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, scopeLine: 8, file: !20, scope: !2, type: !4)
!2 = !DIFile(filename: "bar.c", directory: "/tmp/")
!3 = distinct !DICompileUnit(language: DW_LANG_C89, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", isOptimized: true, emissionKind: FullDebug, file: !20, enums: !21, retainedTypes: !21, subprograms: !{!1, !10})
!4 = !DISubroutineType(types: !5)
!5 = !{null, !6}
!6 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!7 = !DILocation(line: 8, scope: !1)
!8 = !DILocation(line: 9, scope: !1)
!9 = !DILocalVariable(name: "x", line: 4, arg: 1, scope: !10, file: !2, type: !6)
!10 = distinct !DISubprogram(name: "bar", linkageName: "bar", line: 4, isLocal: true, isDefinition: true, virtualIndex: 6, isOptimized: false, scopeLine: 4, file: !20, scope: !2, type: !11)
!11 = !DISubroutineType(types: !12)
!12 = !{null, !6, !13, !14}
!13 = !DIBasicType(tag: DW_TAG_base_type, name: "long int", size: 64, align: 64, encoding: DW_ATE_signed)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, file: !20, scope: !2, baseType: null)
!15 = !DILocation(line: 4, scope: !10, inlinedAt: !8)
!16 = !DILocalVariable(name: "y", line: 4, arg: 2, scope: !10, file: !2, type: !13)
!17 = !DILocalVariable(name: "z", line: 4, arg: 3, scope: !10, file: !2, type: !14)
!18 = !DILocation(line: 5, scope: !10, inlinedAt: !8)
!19 = !DILocation(line: 10, scope: !1)
!20 = !DIFile(filename: "bar.c", directory: "/tmp/")
!21 = !{}
!22 = !{i32 1, !"Debug Info Version", i32 3}
