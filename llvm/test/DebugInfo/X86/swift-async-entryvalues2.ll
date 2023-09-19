; RUN: llc -experimental-debug-variable-locations=true -stop-after=livedebugvalues -verify-machineinstrs -march=x86-64 -o - %s | FileCheck %s
;
; CHECK: DBG_VALUE $r14, 0, {{.*}}, !DIExpression(DW_OP_LLVM_entry_value, 1, DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 8, DW_OP_deref)
; CHECK: DBG_VALUE $r14, 0, {{.*}}, !DIExpression(DW_OP_LLVM_entry_value, 1, DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 16)
; CHECK: DBG_VALUE $r14, 0, {{.*}}, !DIExpression(DW_OP_LLVM_entry_value, 1, DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 16)
; CHECK: DBG_VALUE $r14, 0, {{.*}}, !DIExpression(DW_OP_LLVM_entry_value, 1, DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 8, DW_OP_deref)
; CHECK-NOT: DBG_VALUE
source_filename = "_Concurrency.ll"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

%swift.opaque = type opaque
%swift.type = type { i64 }
%TScG8IteratorV = type <{ %TScG, %TSb }>
%TScG = type <{ i8* }>
%TSb = type <{ i1 }>
%swift.context = type { %swift.context*, void (%swift.context*)* }

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #0

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: argmemonly nounwind
declare extern_weak swiftcc void @swift_task_dealloc(i8*) local_unnamed_addr #2

; Function Attrs: nounwind
define hidden swifttailcc void @"$sScG8IteratorV4nextxSgyYaFTY2_"(i8* swiftasync %0) #3 !dbg !22 {
entryresume.2:
  call void @llvm.dbg.declare(metadata i8* %0, metadata !44, metadata !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 8, DW_OP_deref)), !dbg !45
  call void @llvm.dbg.declare(metadata i8* %0, metadata !44, metadata !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 8, DW_OP_deref)), !dbg !45
  call void @llvm.dbg.declare(metadata i8* %0, metadata !44, metadata !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 8, DW_OP_deref)), !dbg !45
  call void @llvm.dbg.declare(metadata i8* %0, metadata !39, metadata !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 16)), !dbg !46
  %async.ctx.frameptr1 = getelementptr inbounds i8, i8* %0, i64 16, !dbg !47
  %.reload.addr46 = getelementptr inbounds i8, i8* %0, i64 56, !dbg !51
  %1 = bitcast i8* %.reload.addr46 to %swift.opaque**, !dbg !51
  %.reload472 = load %swift.opaque*, %swift.opaque** %1, align 8, !dbg !51
  %ChildTaskResult.reload.addr38 = getelementptr inbounds i8, i8* %0, i64 32, !dbg !51
  %2 = bitcast i8* %ChildTaskResult.reload.addr38 to %swift.type**, !dbg !51
  %ChildTaskResult.reload39 = load %swift.type*, %swift.type** %2, align 8, !dbg !51
  %3 = getelementptr inbounds %swift.type, %swift.type* %ChildTaskResult.reload39, i64 -1, !dbg !51
  %4 = bitcast %swift.type* %3 to i8***, !dbg !51
  %ChildTaskResult.valueWitnesses = load i8**, i8*** %4, align 8, !dbg !51, !invariant.load !18, !dereferenceable !52
  %5 = getelementptr inbounds i8*, i8** %ChildTaskResult.valueWitnesses, i64 6, !dbg !51
  %6 = bitcast i8** %5 to i32 (%swift.opaque*, i32, %swift.type*)**, !dbg !51
  %7 = load i32 (%swift.opaque*, i32, %swift.type*)*, i32 (%swift.opaque*, i32, %swift.type*)** %6, align 8, !dbg !51, !invariant.load !18
  %8 = tail call i32 %7(%swift.opaque* noalias %.reload472, i32 1, %swift.type* %ChildTaskResult.reload39) #4, !dbg !51
  %.not = icmp eq i32 %8, 1
  br i1 %.not, label %.from.16, label %.from.14

.from.14:                                         ; preds = %entryresume.2
  %9 = bitcast i8* %async.ctx.frameptr1 to %swift.opaque**, !dbg !53
  %.reload203 = load %swift.opaque*, %swift.opaque** %9, align 8, !dbg !53
  %10 = getelementptr inbounds i8*, i8** %ChildTaskResult.valueWitnesses, i64 4, !dbg !55
  %11 = bitcast i8** %10 to %swift.opaque* (%swift.opaque*, %swift.opaque*, %swift.type*)**, !dbg !55
  %12 = load %swift.opaque* (%swift.opaque*, %swift.opaque*, %swift.type*)*, %swift.opaque* (%swift.opaque*, %swift.opaque*, %swift.type*)** %11, align 8, !dbg !55, !invariant.load !18
  %13 = tail call %swift.opaque* %12(%swift.opaque* noalias %.reload203, %swift.opaque* noalias %.reload472, %swift.type* nonnull %ChildTaskResult.reload39) #5, !dbg !55
  %14 = getelementptr inbounds i8*, i8** %ChildTaskResult.valueWitnesses, i64 7, !dbg !53
  %15 = bitcast i8** %14 to void (%swift.opaque*, i32, i32, %swift.type*)**, !dbg !53
  %16 = load void (%swift.opaque*, i32, i32, %swift.type*)*, void (%swift.opaque*, i32, i32, %swift.type*)** %15, align 8, !dbg !53, !invariant.load !18
  tail call void %16(%swift.opaque* noalias %.reload203, i32 0, i32 1, %swift.type* nonnull %ChildTaskResult.reload39) #5, !dbg !53
  br label %AfterMustTailCall.Before.CoroEnd, !dbg !56

.from.16:                                         ; preds = %entryresume.2
  %.valueWitnesses.reload.addr = getelementptr inbounds i8, i8* %0, i64 48, !dbg !57
  %17 = bitcast i8* %.valueWitnesses.reload.addr to i8***, !dbg !57
  %.valueWitnesses.reload = load i8**, i8*** %17, align 8, !dbg !57
  %.reload.addr43 = getelementptr inbounds i8, i8* %0, i64 40, !dbg !57
  %18 = bitcast i8* %.reload.addr43 to %swift.type**, !dbg !57
  %.reload44 = load %swift.type*, %swift.type** %18, align 8, !dbg !57
  %.reload.addr24 = getelementptr inbounds i8, i8* %0, i64 24, !dbg !57
  %19 = bitcast i8* %.reload.addr24 to %TScG8IteratorV**, !dbg !57
  %.reload25 = load %TScG8IteratorV*, %TScG8IteratorV** %19, align 8, !dbg !57
  %20 = bitcast i8* %async.ctx.frameptr1 to %swift.opaque**, !dbg !57
  %.reload224 = load %swift.opaque*, %swift.opaque** %20, align 8, !dbg !57
  %.finished._value18 = getelementptr inbounds %TScG8IteratorV, %TScG8IteratorV* %.reload25, i64 0, i32 1, i32 0, !dbg !57
  %21 = getelementptr inbounds i8*, i8** %.valueWitnesses.reload, i64 1, !dbg !58
  %22 = bitcast i8** %21 to void (%swift.opaque*, %swift.type*)**, !dbg !58
  %23 = load void (%swift.opaque*, %swift.type*)*, void (%swift.opaque*, %swift.type*)** %22, align 8, !dbg !58, !invariant.load !18
  tail call void %23(%swift.opaque* noalias %.reload472, %swift.type* %.reload44) #5, !dbg !58
  %24 = bitcast i1* %.finished._value18 to i8*, !dbg !60
  store i8 1, i8* %24, align 8, !dbg !60
  %25 = getelementptr inbounds i8*, i8** %ChildTaskResult.valueWitnesses, i64 7, !dbg !62
  %26 = bitcast i8** %25 to void (%swift.opaque*, i32, i32, %swift.type*)**, !dbg !62
  %27 = load void (%swift.opaque*, i32, i32, %swift.type*)*, void (%swift.opaque*, i32, i32, %swift.type*)** %26, align 8, !dbg !62, !invariant.load !18
  tail call void %27(%swift.opaque* noalias %.reload224, i32 1, i32 1, %swift.type* nonnull %ChildTaskResult.reload39) #5, !dbg !62
  br label %AfterMustTailCall.Before.CoroEnd, !dbg !63

AfterMustTailCall.Before.CoroEnd:                 ; preds = %.from.16, %.from.14
  %28 = bitcast i8* %.reload.addr46 to i8**, !dbg !51
  %.reload51 = load i8*, i8** %28, align 8, !dbg !64
  tail call void @llvm.lifetime.end.p0i8(i64 -1, i8* %.reload51), !dbg !64
  tail call swiftcc void @swift_task_dealloc(i8* %.reload51) #5, !dbg !64
  %29 = getelementptr inbounds i8, i8* %0, i64 8, !dbg !64
  %30 = bitcast i8* %29 to void (%swift.context*)**, !dbg !64
  %31 = load void (%swift.context*)*, void (%swift.context*)** %30, align 8, !dbg !64
  %32 = bitcast i8* %0 to %swift.context*, !dbg !64
  musttail call swifttailcc void %31(%swift.context* swiftasync %32) #5, !dbg !65
  ret void, !dbg !65
}

attributes #0 = { argmemonly nocallback nofree nosync nounwind willreturn }
attributes #1 = { nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { argmemonly nounwind }
attributes #3 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="core2" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+ssse3,+x87" "tune-cpu"="generic" }
attributes #4 = { nounwind readonly }
attributes #5 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15}
!llvm.dbg.cu = !{!16}
!swift.module.flags = !{!19}
!llvm.linker.options = !{!20, !21}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 3]}
!1 = !{i32 1, !"Objective-C Version", i32 2}
!2 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!3 = !{i32 1, !"Objective-C Image Info Section", !"__DATA,__objc_imageinfo,regular,no_dead_strip"}
!4 = !{i32 1, !"Objective-C Garbage Collection", i8 0}
!5 = !{i32 1, !"Objective-C Class Properties", i32 64}
!6 = !{i32 7, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{i32 7, !"PIC Level", i32 2}
!10 = !{i32 7, !"uwtable", i32 2}
!11 = !{i32 7, !"frame-pointer", i32 2}
!12 = !{i32 1, !"Swift Version", i32 7}
!13 = !{i32 1, !"Swift ABI Version", i32 7}
!14 = !{i32 1, !"Swift Major Version", i8 5}
!15 = !{i32 1, !"Swift Minor Version", i8 8}
!16 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !17, producer: "Swift version 5.8-dev (LLVM 8ee67b87b160b1d, Swift c4a0e0b8a1e003f)", isOptimized: true, runtimeVersion: 5, emissionKind: FullDebug, globals: !18, imports: !18, sysroot: "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX12.3.sdk", sdk: "MacOSX12.3.sdk")
!17 = !DIFile(filename: "/swift/stdlib/public/Concurrency/Actor.swift", directory: "/")
!18 = !{}
!19 = !{!"standard-library", i1 false}
!20 = !{!"-lswiftCore"}
!21 = !{!"-lobjc"}
!22 = distinct !DISubprogram(name: "next", linkageName: "$sScG8IteratorV4nextxSgyYaFTY2_", scope: !24, file: !23, line: 765, type: !26, scopeLine: 767, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !16, retainedNodes: !38)
!23 = !DIFile(filename: "swift/stdlib/public/Concurrency/TaskGroup.swift", directory: "/")
!24 = !DICompositeType(tag: DW_TAG_structure_type, name: "Iterator", scope: !25, file: !23, flags: DIFlagFwdDecl, runtimeLang: DW_LANG_Swift)
!25 = !DIModule(scope: null, name: "_Concurrency", includePath: "/swift/stdlib/public/Concurrency")
!26 = !DISubroutineType(types: !27)
!27 = !{!28, !37}
!28 = !DICompositeType(tag: DW_TAG_structure_type, scope: !30, file: !29, elements: !31, runtimeLang: DW_LANG_Swift)
!29 = !DIFile(filename: "/lib/swift/macosx/Swift.swiftmodule/x86_64-apple-macos.swiftmodule", directory: "/")
!30 = !DIModule(scope: null, name: "Swift", configMacros: "\22-DSWIFT_STDLIB_HAS_ENVIRON\22", includePath: "/lib/swift/macosx/Swift.swiftmodule/x86_64-apple-macos.swiftmodule")
!31 = !{!32}
!32 = !DIDerivedType(tag: DW_TAG_member, scope: !30, file: !29, baseType: !33)
!33 = !DICompositeType(tag: DW_TAG_structure_type, name: "Optional", scope: !30, file: !29, flags: DIFlagFwdDecl, runtimeLang: DW_LANG_Swift, templateParams: !34, identifier: "$sxSgD")
!34 = !{!35}
!35 = !DITemplateTypeParameter(type: !36)
!36 = !DICompositeType(tag: DW_TAG_structure_type, name: "$sxD", file: !17, runtimeLang: DW_LANG_Swift, identifier: "$sxD")
!37 = !DICompositeType(tag: DW_TAG_structure_type, name: "Iterator", scope: !25, file: !23, size: 72, elements: !18, runtimeLang: DW_LANG_Swift, identifier: "$sScG8IteratorVyx_GD")
!38 = !{!39, !44}
!39 = !DILocalVariable(name: "$\CF\84_0_0", scope: !22, file: !17, type: !40, flags: DIFlagArtificial)
!40 = !DIDerivedType(tag: DW_TAG_typedef, name: "ChildTaskResult", scope: !42, file: !41, baseType: !43)
!41 = !DIFile(filename: "<compiler-generated>", directory: "")
!42 = !DIModule(scope: null, name: "Builtin", configMacros: "\22-DSWIFT_STDLIB_HAS_ENVIRON\22")
!43 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "$sBpD", baseType: null, size: 64)
!44 = !DILocalVariable(name: "self", arg: 1, scope: !22, file: !17, line: 765, type: !37, flags: DIFlagArtificial)
!45 = !DILocation(line: 765, column: 26, scope: !22)
!46 = !DILocation(line: 0, scope: !22)
!47 = !DILocation(line: 0, scope: !48)
!48 = !DILexicalBlockFile(scope: !49, file: !41, discriminator: 0)
!49 = distinct !DILexicalBlock(scope: !50, file: !23, line: 767, column: 7)
!50 = distinct !DILexicalBlock(scope: !22, file: !23, line: 765, column: 51)
!51 = !DILocation(line: 767, column: 39, scope: !49)
!52 = !{i64 96}
!53 = !DILocation(line: 771, column: 14, scope: !54)
!54 = distinct !DILexicalBlock(scope: !49, file: !23, line: 767, column: 33)
!55 = !DILocation(line: 767, column: 39, scope: !54)
!56 = !DILocation(line: 771, column: 7, scope: !54)
!57 = !DILocation(line: 766, column: 14, scope: !50)
!58 = !DILocation(line: 0, scope: !59)
!59 = !DILexicalBlockFile(scope: !54, file: !41, discriminator: 0)
!60 = !DILocation(line: 768, column: 18, scope: !61)
!61 = distinct !DILexicalBlock(scope: !50, file: !23, line: 767, column: 51)
!62 = !DILocation(line: 769, column: 16, scope: !61)
!63 = !DILocation(line: 769, column: 9, scope: !61)
!64 = !DILocation(line: 772, column: 5, scope: !54)
!65 = !DILocation(line: 0, scope: !66, inlinedAt: !68)
!66 = distinct !DISubprogram(linkageName: "$sScG8IteratorV4nextxSgyYaF", scope: !25, file: !41, type: !67, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !16, retainedNodes: !18)
!67 = !DISubroutineType(types: null)
!68 = distinct !DILocation(line: 772, column: 5, scope: !54)
