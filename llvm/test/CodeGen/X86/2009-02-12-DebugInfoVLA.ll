; RUN: llc < %s
; RUN: llc < %s -march=x86-64 -verify-machineinstrs | FileCheck %s
; PR3538
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9"
define signext i8 @foo(i8* %s1) nounwind ssp {

; Make sure we generate:
;  movq	-40(%rbp), %rsp
; Instead of:
;  movq	-40(%rbp), %rax
;  movq	%rax, %rsp

; CHECK-LABEL: @foo
; CHECK: movq	-40(%rbp), %rsp

entry:
  %s1_addr = alloca i8*                           ; <i8**> [#uses=2]
  %retval = alloca i32                            ; <i32*> [#uses=2]
  %saved_stack.1 = alloca i8*                     ; <i8**> [#uses=2]
  %0 = alloca i32                                 ; <i32*> [#uses=2]
  %str.0 = alloca [0 x i8]*                       ; <[0 x i8]**> [#uses=3]
  %1 = alloca i64                                 ; <i64*> [#uses=2]
  %2 = alloca i64                                 ; <i64*> [#uses=1]
  %3 = alloca i64                                 ; <i64*> [#uses=6]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata i8** %s1_addr, metadata !0, metadata !DIExpression()), !dbg !7
  store i8* %s1, i8** %s1_addr
  call void @llvm.dbg.declare(metadata [0 x i8]** %str.0, metadata !8, metadata !DIExpression()), !dbg !7
  %4 = call i8* @llvm.stacksave(), !dbg !7        ; <i8*> [#uses=1]
  store i8* %4, i8** %saved_stack.1, align 8, !dbg !7
  %5 = load i8*, i8** %s1_addr, align 8, !dbg !13      ; <i8*> [#uses=1]
  %6 = call i64 @strlen(i8* %5) nounwind readonly, !dbg !13 ; <i64> [#uses=1]
  %7 = add i64 %6, 1, !dbg !13                    ; <i64> [#uses=1]
  store i64 %7, i64* %3, align 8, !dbg !13
  %8 = load i64, i64* %3, align 8, !dbg !13            ; <i64> [#uses=1]
  %9 = sub nsw i64 %8, 1, !dbg !13                ; <i64> [#uses=0]
  %10 = load i64, i64* %3, align 8, !dbg !13           ; <i64> [#uses=1]
  %11 = mul i64 %10, 8, !dbg !13                  ; <i64> [#uses=0]
  %12 = load i64, i64* %3, align 8, !dbg !13           ; <i64> [#uses=1]
  store i64 %12, i64* %2, align 8, !dbg !13
  %13 = load i64, i64* %3, align 8, !dbg !13           ; <i64> [#uses=1]
  %14 = mul i64 %13, 8, !dbg !13                  ; <i64> [#uses=0]
  %15 = load i64, i64* %3, align 8, !dbg !13           ; <i64> [#uses=1]
  store i64 %15, i64* %1, align 8, !dbg !13
  %16 = load i64, i64* %1, align 8, !dbg !13           ; <i64> [#uses=1]
  %17 = trunc i64 %16 to i32, !dbg !13            ; <i32> [#uses=1]
  %18 = alloca i8, i32 %17, !dbg !13              ; <i8*> [#uses=1]
  %19 = bitcast i8* %18 to [0 x i8]*, !dbg !13    ; <[0 x i8]*> [#uses=1]
  store [0 x i8]* %19, [0 x i8]** %str.0, align 8, !dbg !13
  %20 = load [0 x i8]*, [0 x i8]** %str.0, align 8, !dbg !15 ; <[0 x i8]*> [#uses=1]
  %21 = getelementptr inbounds [0 x i8], [0 x i8]* %20, i64 0, i64 0, !dbg !15 ; <i8*> [#uses=1]
  store i8 0, i8* %21, align 1, !dbg !15
  %22 = load [0 x i8]*, [0 x i8]** %str.0, align 8, !dbg !16 ; <[0 x i8]*> [#uses=1]
  %23 = getelementptr inbounds [0 x i8], [0 x i8]* %22, i64 0, i64 0, !dbg !16 ; <i8*> [#uses=1]
  %24 = load i8, i8* %23, align 1, !dbg !16           ; <i8> [#uses=1]
  %25 = sext i8 %24 to i32, !dbg !16              ; <i32> [#uses=1]
  store i32 %25, i32* %0, align 4, !dbg !16
  %26 = load i8*, i8** %saved_stack.1, align 8, !dbg !16 ; <i8*> [#uses=1]
  call void @llvm.stackrestore(i8* %26), !dbg !16
  %27 = load i32, i32* %0, align 4, !dbg !16           ; <i32> [#uses=1]
  store i32 %27, i32* %retval, align 4, !dbg !16
  br label %return, !dbg !16

return:                                           ; preds = %entry
  %retval1 = load i32, i32* %retval, !dbg !16          ; <i32> [#uses=1]
  %retval12 = trunc i32 %retval1 to i8, !dbg !16  ; <i8> [#uses=1]
  ret i8 %retval12, !dbg !16
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare i8* @llvm.stacksave() nounwind

declare i64 @strlen(i8*) nounwind readonly

declare void @llvm.stackrestore(i8*) nounwind

!0 = !DILocalVariable(name: "s1", line: 2, arg: 1, scope: !1, file: !2, type: !6)
!1 = distinct !DISubprogram(name: "foo", linkageName: "foo", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, scope: !2, type: !3)
!2 = distinct !DICompileUnit(language: DW_LANG_C89, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", isOptimized: true, emissionKind: 0, file: !17, enums: !18, retainedTypes: !18)
!3 = !DISubroutineType(types: !4)
!4 = !{!5, !6}
!5 = !DIBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, scope: !2, baseType: !5)
!7 = !DILocation(line: 2, scope: !1)
!8 = !DILocalVariable(name: "str.0", line: 3, scope: !1, file: !2, type: !9)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial, scope: !2, baseType: !10)
!10 = !DICompositeType(tag: DW_TAG_array_type, size: 8, align: 8, scope: !2, baseType: !5, elements: !11)
!11 = !{!12}
!12 = !DISubrange(count: 1)
!13 = !DILocation(line: 3, scope: !14)
!14 = distinct !DILexicalBlock(line: 0, column: 0, file: !17, scope: !1)
!15 = !DILocation(line: 4, scope: !14)
!16 = !DILocation(line: 5, scope: !14)
!17 = !DIFile(filename: "vla.c", directory: "/tmp/")
!18 = !{i32 0}
