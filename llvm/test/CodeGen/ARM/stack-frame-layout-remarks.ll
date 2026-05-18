; Test remark output for stack-frame-layout

; ensure basic output works
; RUN: llc -mtriple=arm-eabi -O1 -pass-remarks-analysis=stack-frame-layout < %s 2>&1 >/dev/null | FileCheck %s

; check additional slots are displayed when stack is not optimized
; RUN: llc -mtriple=arm-eabi -O0  -pass-remarks-analysis=stack-frame-layout < %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NO_COLORING

; check more complex cases
; RUN: llc %s  -pass-remarks-analysis=stack-frame-layout -o /dev/null --mtriple=arm -mcpu=cortex-m1 2>&1 | FileCheck %s --check-prefix=BOTH --check-prefix=DEBUG

; check output without debug info
; RUN: opt %s -passes=strip -S | llc   -pass-remarks-analysis=stack-frame-layout -o /dev/null --mtriple=arm -mcpu=cortex-m1 2>&1 | FileCheck %s --check-prefix=BOTH --check-prefix=STRIPPED

target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1
declare i32 @printf(ptr, ...)

; CHECK: Function: stackSizeWarning
; CHECK: [SP-4]{{.*}}Spill{{.*}}4{{.*}}4
; CHECK: [SP-96]{{.*}}16{{.*}}80
; CHECK:    buffer @ frame-diags.c:30
; NO_COLORING: [SP-176]{{.*}}16{{.*}}80
; CHECK:    buffer2 @ frame-diags.c:33

; BOTH: Function: stackSizeWarning
; BOTH: [SP-4]{{.*}}Spill{{.*}}4{{.*}}4
; BOTH: [SP-8]{{.*}}Spill{{.*}}4{{.*}}4
; BOTH: [SP-12]{{.*}}Spill{{.*}}4{{.*}}4
; BOTH: [SP-16]{{.*}}Spill{{.*}}4{{.*}}4
; BOTH: [SP-96]{{.*}}16{{.*}}80
; DEBUG: buffer @ frame-diags.c:30
; STRIPPED-NOT: buffer @ frame-diags.c:30
; BOTH: [SP-176]{{.*}}16{{.*}}80
; DEBUG: buffer2 @ frame-diags.c:33
; STRIPPED-NOT: buffer2 @ frame-diags.c:33
define void @stackSizeWarning() {
entry:
  %buffer = alloca [80 x i8], align 16
  %buffer2 = alloca [80 x i8], align 16
  call void @llvm.dbg.declare(metadata ptr %buffer, metadata !25, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata ptr %buffer2, metadata !31, metadata !DIExpression()), !dbg !40
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

; BOTH: Function: cleanup_array
; BOTH:  [SP-8]{{.+}}8{{.+}}4
; DEBUG: a @ dot.c:13
; STRIPPED-NOT: a @ dot.c:13
define void @cleanup_array(ptr %0) #3 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  call void @llvm.dbg.declare(metadata ptr %2, metadata !41, metadata !DIExpression()), !dbg !46
  ret void
}

; BOTH: Function: cleanup_result
; BOTH:  [SP-8]{{.+}}8{{.+}}4
; DEBUG: res @ dot.c:21
; STRIPPED-NOT: res @ dot.c:21
define void @cleanup_result(ptr %0) #3 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  call void @llvm.dbg.declare(metadata ptr %2, metadata !47, metadata !DIExpression()), !dbg !51
  ret void
}

; BOTH: Function: do_work
; BOTH:  [SP-4]{{.+}}4{{.+}}4
; BOTH:  [SP-8]{{.+}}8{{.+}}4
; DEBUG: A @ dot.c:32
; STRIPPED-NOT: A @ dot.c:32
; BOTH:  [SP-16]{{.+}}8{{.+}}4
; DEBUG: B @ dot.c:32
; STRIPPED-NOT: B @ dot.c:32
; BOTH:  [SP-24]{{.+}}8{{.+}}4
; DEBUG: out @ dot.c:32
; STRIPPED-NOT: out @ dot.c:32
; BOTH:  [SP-28]{{.+}}4{{.+}}4
; DEBUG: len @ dot.c:37
; STRIPPED-NOT: len @ dot.c:37
; BOTH:  [SP-32]{{.+}}8{{.+}}4
; DEBUG: AB @ dot.c:38
; STRIPPED-NOT: AB @ dot.c:38
; BOTH:  [SP-36]{{.+}}4{{.+}}4
; DEBUG: sum @ dot.c:54
; STRIPPED-NOT: sum @ dot.c:54
; BOTH:  [SP-40]{{.+}}4{{.+}}4
; DEBUG: i @ dot.c:55
; STRIPPED-NOT: i @ dot.c:55
define i32 @do_work(ptr %0, ptr %1, ptr %2) #3 {
  %4 = alloca i32, align 4
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca i32, align 4
  %9 = alloca ptr, align 8
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  store ptr %0, ptr %5, align 8
  call void @llvm.dbg.declare(metadata ptr %5, metadata !52, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.declare(metadata ptr %6, metadata !57, metadata !DIExpression()), !dbg !58
  store ptr %2, ptr %7, align 8
  call void @llvm.dbg.declare(metadata ptr %7, metadata !59, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.declare(metadata ptr %8, metadata !61, metadata !DIExpression()), !dbg !63
  call void @llvm.dbg.declare(metadata ptr %9, metadata !64, metadata !DIExpression()), !dbg !65
  store ptr null, ptr %9, align 8
  store ptr null, ptr null, align 8
  store i32 0, ptr %9, align 8
  %12 = load i32, ptr %8, align 4
  store i32 %12, ptr null, align 8
  call void @llvm.dbg.declare(metadata ptr %10, metadata !66, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.declare(metadata ptr %11, metadata !68, metadata !DIExpression()), !dbg !70
  store i32 0, ptr %11, align 4
  br label %13

13:                                               ; preds = %16, %3
  %14 = load i32, ptr %11, align 4
  %15 = icmp slt i32 %14, 0
  br i1 %15, label %16, label %18

16:                                               ; preds = %13
  %17 = load i32, ptr %6, align 4
  store i32 %17, ptr null, align 4
  br label %13

18:                                               ; preds = %13
  store i32 0, ptr %4, align 4
  ret i32 0
}

; BOTH: Function: gen_array
; BOTH:  [SP-8]{{.+}}8{{.+}}4
; BOTH:  [SP-12]{{.+}}4{{.+}}4
; DEBUG: size @ dot.c:62
; STRIPPED-NOT: size @ dot.c:65
; BOTH:  [SP-16]{{.+}}8{{.+}}4
; DEBUG: res @ dot.c:65
; STRIPPED-NOT: res @ dot.c:65
; BOTH:  [SP-20]{{.+}}4{{.*}}4
; DEBUG: i @ dot.c:69
; STRIPPED-NOT: i @ dot.c:69
define ptr @gen_array(i32 %0) #3 {
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  store i32 %0, ptr %3, align 4
  call void @llvm.dbg.declare(metadata ptr %3, metadata !71, metadata !DIExpression()), !dbg !75
  call void @llvm.dbg.declare(metadata ptr %4, metadata !76, metadata !DIExpression()), !dbg !77
  store ptr null, ptr %4, align 8
  call void @llvm.dbg.declare(metadata ptr %5, metadata !78, metadata !DIExpression()), !dbg !80
  store i32 0, ptr %5, align 4
  ret ptr null
}


; BOTH: Function: caller
; BOTH: [SP-4]{{.*}}Spill{{.*}}4{{.*}}4
; BOTH: [SP-8]{{.*}}Spill{{.*}}4{{.*}}4
; BOTH: [SP-12]{{.*}}Spill{{.*}}4{{.*}}4
; BOTH: [SP-16]{{.*}}Spill{{.*}}4{{.*}}4
; BOTH: [SP-20]{{.*}}4{{.*}}4
; BOTH: [SP-24]{{.*}}4{{.*}}4
; DEBUG: size @ dot.c:77
; STRIPPED-NOT: size @ dot.c:77
; BOTH: [SP-32]{{.*}}8{{.*}}4
; DEBUG: A @ dot.c:78
; STRIPPED-NOT: A @ dot.c:78
; BOTH: [SP-40]{{.*}}8{{.*}}4
; DEBUG: B @ dot.c:79
; STRIPPED-NOT: B @ dot.c:79
; BOTH: [SP-48]{{.*}}8{{.*}}4
; DEBUG: res @ dot.c:80
; STRIPPED-NOT: res @ dot.c:80
; BOTH: [SP-52]{{.*}}4{{.*}}4
; DEBUG: ret @ dot.c:81
; STRIPPED-NOT: ret @ dot.c:81
; BOTH: [SP-56]{{.*}}4{{.*}}4
; DEBUG: err @ dot.c:83
; STRIPPED-NOT: err @ dot.c:83
define i32 @caller() #1 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata ptr %2, metadata !81, metadata !DIExpression()), !dbg !85
  call void @llvm.dbg.declare(metadata ptr %3, metadata !86, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.declare(metadata ptr %4, metadata !88, metadata !DIExpression()), !dbg !89
  store ptr null, ptr %4, align 8
  call void @llvm.dbg.declare(metadata ptr %5, metadata !90, metadata !DIExpression()), !dbg !91
  call void @llvm.dbg.declare(metadata ptr %6, metadata !92, metadata !DIExpression()), !dbg !93
  call void @llvm.dbg.declare(metadata ptr %7, metadata !94, metadata !DIExpression()), !dbg !95
  %8 = call i32 @do_work(ptr %3, ptr null, ptr null)
  store i32 0, ptr %6, align 4
  store i32 0, ptr %1, align 4
  call void @cleanup_result(ptr %5)
  ret i32 0
}

; test29b: An array of [5 x i8] and a requested ssp-buffer-size of 5.
; Requires protector.
; Function Attrs: ssp stack-protector-buffer-size=5
; BOTH: Function: test29b
; BOTH:  [SP-4]{{.+}}Spill{{.*}}4{{.+}}4
; BOTH:  [SP-8]{{.+}}Spill{{.*}}4{{.+}}4
; BOTH:  [SP-12]{{.+}}Protector{{.*}}4{{.+}}4
; BOTH:  [SP-20]{{.+}}4{{.+}}5
define i32 @test29b() #2 {
entry:
  %test = alloca [5 x i8], align 1
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %test)
  ret i32 %call
}


; uselistorder directives
uselistorder ptr @llvm.dbg.declare, { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 18 }

attributes #0 = { nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #1 = { "frame-pointer"="all" }
attributes #2 = { ssp "stack-protector-buffer-size"="5" "frame-pointer"="all" }
attributes #3 = { "frame-pointer"="none" }

!llvm.dbg.cu = !{!0, !2}
!llvm.module.flags = !{!18, !19, !20, !21, !22, !23, !24}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "frame-diags.c", directory: "")
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "dot.c", directory: "")
!4 = !{!5, !6, !10, !13}
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Array", file: !3, line: 3, size: 128, elements: !8)
!8 = !{!9, !12}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "data", scope: !7, file: !3, line: 4, baseType: !10, size: 64)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DIDerivedType(tag: DW_TAG_member, name: "size", scope: !7, file: !3, line: 5, baseType: !11, size: 32, offset: 64)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!14 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Result", file: !3, line: 8, size: 128, elements: !15)
!15 = !{!16, !17}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "data", scope: !14, file: !3, line: 9, baseType: !6, size: 64)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "sum", scope: !14, file: !3, line: 10, baseType: !11, size: 32, offset: 64)
!18 = !{i32 7, !"Dwarf Version", i32 5}
!19 = !{i32 2, !"Debug Info Version", i32 3}
!20 = !{i32 1, !"wchar_size", i32 4}
!21 = !{i32 8, !"PIC Level", i32 2}
!22 = !{i32 7, !"PIE Level", i32 2}
!23 = !{i32 7, !"uwtable", i32 2}
!24 = !{i32 7, !"frame-pointer", i32 2}
!25 = !DILocalVariable(name: "buffer", scope: !26, file: !1, line: 30, type: !32)
!26 = distinct !DILexicalBlock(scope: !27, file: !1, line: 29, column: 3)
!27 = distinct !DISubprogram(name: "stackSizeWarning", scope: !1, file: !1, line: 28, type: !28, scopeLine: 28, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !30)
!28 = !DISubroutineType(types: !29)
!29 = !{null}
!30 = !{!25, !31, !36, !37}
!31 = !DILocalVariable(name: "buffer2", scope: !27, file: !1, line: 33, type: !32)
!32 = !DICompositeType(tag: DW_TAG_array_type, baseType: !33, size: 640, elements: !34)
!33 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!34 = !{!35}
!35 = !DISubrange(count: 80)
!36 = !DILocalVariable(name: "a", scope: !27, file: !1, line: 34, type: !11)
!37 = !DILocalVariable(name: "b", scope: !27, file: !1, line: 35, type: !38)
!38 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!39 = !DILocation(line: 30, column: 10, scope: !26)
!40 = !DILocation(line: 33, column: 8, scope: !27)
!41 = !DILocalVariable(name: "a", arg: 1, scope: !42, file: !3, line: 13, type: !6)
!42 = distinct !DISubprogram(name: "cleanup_array", scope: !3, file: !3, line: 13, type: !43, scopeLine: 13, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !45)
!43 = !DISubroutineType(types: !44)
!44 = !{null, !6}
!45 = !{}
!46 = !DILocation(line: 13, column: 34, scope: !42)
!47 = !DILocalVariable(name: "res", arg: 1, scope: !48, file: !3, line: 21, type: !13)
!48 = distinct !DISubprogram(name: "cleanup_result", scope: !3, file: !3, line: 21, type: !49, scopeLine: 21, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !45)
!49 = !DISubroutineType(types: !50)
!50 = !{null, !13}
!51 = !DILocation(line: 21, column: 36, scope: !48)
!52 = !DILocalVariable(name: "A", arg: 1, scope: !53, file: !3, line: 32, type: !6)
!53 = distinct !DISubprogram(name: "do_work", scope: !3, file: !3, line: 32, type: !54, scopeLine: 32, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !45)
!54 = !DISubroutineType(types: !55)
!55 = !{!11, !6, !6, !13}
!56 = !DILocation(line: 32, column: 27, scope: !53)
!57 = !DILocalVariable(name: "B", arg: 2, scope: !53, file: !3, line: 32, type: !6)
!58 = !DILocation(line: 32, column: 44, scope: !53)
!59 = !DILocalVariable(name: "out", arg: 3, scope: !53, file: !3, line: 32, type: !13)
!60 = !DILocation(line: 32, column: 62, scope: !53)
!61 = !DILocalVariable(name: "len", scope: !53, file: !3, line: 37, type: !62)
!62 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !11)
!63 = !DILocation(line: 37, column: 13, scope: !53)
!64 = !DILocalVariable(name: "AB", scope: !53, file: !3, line: 38, type: !6)
!65 = !DILocation(line: 38, column: 17, scope: !53)
!66 = !DILocalVariable(name: "sum", scope: !53, file: !3, line: 54, type: !11)
!67 = !DILocation(line: 54, column: 7, scope: !53)
!68 = !DILocalVariable(name: "i", scope: !69, file: !3, line: 55, type: !11)
!69 = distinct !DILexicalBlock(scope: !53, file: !3, line: 55, column: 3)
!70 = !DILocation(line: 55, column: 12, scope: !69)
!71 = !DILocalVariable(name: "size", arg: 1, scope: !72, file: !3, line: 62, type: !11)
!72 = distinct !DISubprogram(name: "gen_array", scope: !3, file: !3, line: 62, type: !73, scopeLine: 62, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !45)
!73 = !DISubroutineType(types: !74)
!74 = !{!6, !11}
!75 = !DILocation(line: 62, column: 29, scope: !72)
!76 = !DILocalVariable(name: "res", scope: !72, file: !3, line: 65, type: !6)
!77 = !DILocation(line: 65, column: 17, scope: !72)
!78 = !DILocalVariable(name: "i", scope: !79, file: !3, line: 69, type: !11)
!79 = distinct !DILexicalBlock(scope: !72, file: !3, line: 69, column: 3)
!80 = !DILocation(line: 69, column: 12, scope: !79)
!81 = !DILocalVariable(name: "size", scope: !82, file: !3, line: 77, type: !62)
!82 = distinct !DISubprogram(name: "caller", scope: !3, file: !3, line: 76, type: !83, scopeLine: 76, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !45)
!83 = !DISubroutineType(types: !84)
!84 = !{!11}
!85 = !DILocation(line: 77, column: 13, scope: !82)
!86 = !DILocalVariable(name: "A", scope: !82, file: !3, line: 78, type: !6)
!87 = !DILocation(line: 78, column: 17, scope: !82)
!88 = !DILocalVariable(name: "B", scope: !82, file: !3, line: 79, type: !6)
!89 = !DILocation(line: 79, column: 17, scope: !82)
!90 = !DILocalVariable(name: "res", scope: !82, file: !3, line: 80, type: !13)
!91 = !DILocation(line: 80, column: 18, scope: !82)
!92 = !DILocalVariable(name: "ret", scope: !82, file: !3, line: 81, type: !11)
!93 = !DILocation(line: 81, column: 7, scope: !82)
!94 = !DILocalVariable(name: "err", scope: !82, file: !3, line: 83, type: !11)
!95 = !DILocation(line: 83, column: 7, scope: !82)
