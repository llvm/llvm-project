; RUN: llc -O0 -filetype=obj < %s | llvm-dwarfdump --name n - | FileCheck %s
;
; CHECK: DW_TAG_formal_parameter
; CHECK-NEXT: DW_AT_location	(DW_OP_entry_value(DW_OP_reg22 W22), DW_OP_deref, DW_OP_plus_uconst 0x18, DW_OP_plus_uconst 0x8)
; CHECK-NEXT: DW_AT_name	("n")

; ModuleID = '/tmp/t.ll'
source_filename = "/tmp/fib.s"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx11.0.0"

%"$s3fib9FibonacciC9fibonacciyS2iYF.Frame" = type { %swift.context*, i64, %T3fib9FibonacciC*, i64, i64, i64, i64, %T3fib9FibonacciC*, i8*, i64, i8*, i64 }
%swift.context = type {}
%T3fib9FibonacciC = type <{ %swift.refcounted, %swift.defaultactor, %TSa }>
%swift.refcounted = type { %swift.type*, i64 }
%swift.type = type { i64 }
%swift.defaultactor = type { [10 x i8*] }
%TSa = type <{ %Ts12_ArrayBufferV }>
%Ts12_ArrayBufferV = type <{ %Ts14_BridgeStorageV }>
%Ts14_BridgeStorageV = type <{ %swift.bridge* }>
%swift.bridge = type opaque
%TSi = type <{ i64 }>
%swift.executor = type {}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #0

define hidden swifttailcc void @"$s3fib9FibonacciC9fibonacciyS2iYFTQ3_"(i8* swiftasync %0, i64 %1) #1 !dbg !59 {
  call void @llvm.dbg.declare(metadata i8* %0, metadata !65, metadata !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 24, DW_OP_plus_uconst, 40)), !dbg !67
  call void @llvm.dbg.declare(metadata i8* %0, metadata !68, metadata !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 24, DW_OP_plus_uconst, 32)), !dbg !69
  call void @llvm.dbg.declare(metadata i8* %0, metadata !70, metadata !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 24, DW_OP_plus_uconst, 24)), !dbg !71
  call void @llvm.dbg.declare(metadata i8* %0, metadata !72, metadata !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 24, DW_OP_plus_uconst, 16)), !dbg !74
  call void @llvm.dbg.declare(metadata i8* %0, metadata !75, metadata !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 24, DW_OP_plus_uconst, 8)), !dbg !76
  %3 = bitcast i8* %0 to i8**, !dbg !77
  %4 = load i8*, i8** %3, align 8, !dbg !77
  %5 = call i8** @llvm.swift.async.context.addr() #3, !dbg !77
  store i8* %4, i8** %5, align 8, !dbg !77
  %6 = getelementptr inbounds i8, i8* %4, i32 24
  %7 = bitcast i8* %6 to %"$s3fib9FibonacciC9fibonacciyS2iYF.Frame"*
  %8 = bitcast %"$s3fib9FibonacciC9fibonacciyS2iYF.Frame"* %7 to i8*
  %9 = alloca %TSi, align 8
  %10 = alloca [32 x i8], align 8
  %11 = alloca %TSi, align 8
  %12 = getelementptr inbounds %"$s3fib9FibonacciC9fibonacciyS2iYF.Frame", %"$s3fib9FibonacciC9fibonacciyS2iYF.Frame"* %7, i32 0, i32 0
  %13 = getelementptr inbounds %"$s3fib9FibonacciC9fibonacciyS2iYF.Frame", %"$s3fib9FibonacciC9fibonacciyS2iYF.Frame"* %7, i32 0, i32 1
  %14 = getelementptr inbounds %"$s3fib9FibonacciC9fibonacciyS2iYF.Frame", %"$s3fib9FibonacciC9fibonacciyS2iYF.Frame"* %7, i32 0, i32 2
  %15 = getelementptr inbounds %"$s3fib9FibonacciC9fibonacciyS2iYF.Frame", %"$s3fib9FibonacciC9fibonacciyS2iYF.Frame"* %7, i32 0, i32 3
  %16 = getelementptr inbounds %"$s3fib9FibonacciC9fibonacciyS2iYF.Frame", %"$s3fib9FibonacciC9fibonacciyS2iYF.Frame"* %7, i32 0, i32 4
  %17 = getelementptr inbounds %"$s3fib9FibonacciC9fibonacciyS2iYF.Frame", %"$s3fib9FibonacciC9fibonacciyS2iYF.Frame"* %7, i32 0, i32 5
  %18 = getelementptr inbounds %"$s3fib9FibonacciC9fibonacciyS2iYF.Frame", %"$s3fib9FibonacciC9fibonacciyS2iYF.Frame"* %7, i32 0, i32 10, !dbg !82
  %19 = load i8*, i8** %18, align 8, !dbg !82
  %20 = getelementptr inbounds %"$s3fib9FibonacciC9fibonacciyS2iYF.Frame", %"$s3fib9FibonacciC9fibonacciyS2iYF.Frame"* %7, i32 0, i32 7, !dbg !82
  %21 = load %T3fib9FibonacciC*, %T3fib9FibonacciC** %20, align 8, !dbg !82
  %22 = bitcast i8* %0 to i8**, !dbg !83
  %23 = load i8*, i8** %22, align 8, !dbg !83
  %24 = call i8** @llvm.swift.async.context.addr() #3, !dbg !83
  store i8* %23, i8** %24, align 8, !dbg !83
  %25 = bitcast i8* %23 to %swift.context*, !dbg !82
  store %swift.context* %25, %swift.context** %12, align 8, !dbg !82
  %26 = getelementptr inbounds %"$s3fib9FibonacciC9fibonacciyS2iYF.Frame", %"$s3fib9FibonacciC9fibonacciyS2iYF.Frame"* %7, i32 0, i32 11, !dbg !82
  store i64 %1, i64* %26, align 8, !dbg !82
  call swiftcc void @swift_task_dealloc(i8* %19) #3, !dbg !82
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %19), !dbg !82
  %27 = bitcast %T3fib9FibonacciC* %21 to %swift.executor*, !dbg !82
  %28 = load %swift.context*, %swift.context** %12, align 8, !dbg !82
  musttail call swifttailcc void @swift_task_switch(%swift.context* swiftasync %28, i8* bitcast (void (i8*)* @"$s3fib9FibonacciC9fibonacciyS2iYFTY4_" to i8*), %swift.executor* %27) #3, !dbg !85
  ret void, !dbg !85
}

declare hidden swifttailcc void @"$s3fib9FibonacciC9fibonacciyS2iYFTY4_"(i8* swiftasync) #1

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

; Function Attrs: nounwind
declare extern_weak swifttailcc void @swift_task_switch(%swift.context*, i8*, %swift.executor*) #3

; Function Attrs: nounwind readnone
declare i8** @llvm.swift.async.context.addr() #4

; Function Attrs: argmemonly nounwind
declare extern_weak swiftcc void @swift_task_dealloc(i8*) #5

attributes #0 = { argmemonly nofree nosync nounwind willreturn }
attributes #1 = { "frame-pointer"="non-leaf" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-a12" "target-features"="+aes,+crc,+crypto,+fp-armv8,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+v8.3a,+zcm,+zcz" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #3 = { nounwind }
attributes #4 = { nounwind readnone }
attributes #5 = { argmemonly nounwind }

!llvm.dbg.cu = !{!0}
!swift.module.flags = !{!12}
!llvm.asan.globals = !{!13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37}
!llvm.module.flags = !{!38, !39, !40, !41, !42, !43, !44, !45, !46, !47, !48, !49, !50, !51, !52, !53, !54}
!llvm.linker.options = !{!55, !56, !57, !58}

!0 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !1, producer: "Swift version 5.5-dev (LLVM 6cafa2dc0b8de57, Swift 9c8f32517768a78)", isOptimized: false, runtimeVersion: 5, emissionKind: FullDebug, enums: !2, imports: !3)
!1 = !DIFile(filename: "/Volumes/Data/swift/llvm-project/../llvm-project/lldb/test/API/lang/swift/async/unwind/backtrace_locals/main.swift", directory: "/Volumes/Data/swift/llvm-project")
!2 = !{}
!3 = !{!4, !6, !8, !10}
!4 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !1, entity: !5, file: !1)
!5 = !DIModule(scope: null, name: "fib", includePath: "/Volumes/Data/swift/llvm-project/../llvm-project/lldb/test/API/lang/swift/async/unwind/backtrace_locals")
!6 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !1, entity: !7, file: !1)
!7 = !DIModule(scope: null, name: "Swift", includePath: "/Volumes/Data/swift/_build/Ninja-RelWithDebInfoAssert/swift-macosx-x86_64/lib/swift/macosx/Swift.swiftmodule/arm64-apple-macos.swiftmodule")
!8 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !1, entity: !9, file: !1)
!9 = !DIModule(scope: null, name: "_Concurrency", includePath: "/Volumes/Data/swift/_build/Ninja-RelWithDebInfoAssert/swift-macosx-x86_64/lib/swift/macosx/_Concurrency.swiftmodule/arm64-apple-macos.swiftmodule")
!10 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !1, entity: !11, file: !1)
!11 = !DIModule(scope: null, name: "SwiftOnoneSupport", includePath: "/Volumes/Data/swift/_build/Ninja-RelWithDebInfoAssert/swift-macosx-x86_64/lib/swift/macosx/SwiftOnoneSupport.swiftmodule/arm64-apple-macos.swiftmodule")
!12 = !{!"standard-library", i1 false}
!13 = distinct !{null, null, null, i1 false, i1 true}
!14 = distinct !{null, null, null, i1 false, i1 true}
!15 = distinct !{null, null, null, i1 false, i1 true}
!16 = distinct !{null, null, null, i1 false, i1 true}
!17 = distinct !{null, null, null, i1 false, i1 true}
!18 = distinct !{null, null, null, i1 false, i1 true}
!19 = distinct !{null, null, null, i1 false, i1 true}
!20 = distinct !{null, null, null, i1 false, i1 true}
!21 = distinct !{null, null, null, i1 false, i1 true}
!22 = distinct !{null, null, null, i1 false, i1 true}
!23 = distinct !{null, null, null, i1 false, i1 true}
!24 = distinct !{null, null, null, i1 false, i1 true}
!25 = distinct !{null, null, null, i1 false, i1 true}
!26 = distinct !{null, null, null, i1 false, i1 true}
!27 = distinct !{null, null, null, i1 false, i1 true}
!28 = distinct !{null, null, null, i1 false, i1 true}
!29 = distinct !{null, null, null, i1 false, i1 true}
!30 = distinct !{null, null, null, i1 false, i1 true}
!31 = distinct !{null, null, null, i1 false, i1 true}
!32 = distinct !{null, null, null, i1 false, i1 true}
!33 = distinct !{null, null, null, i1 false, i1 true}
!34 = distinct !{null, null, null, i1 false, i1 true}
!35 = distinct !{null, null, null, i1 false, i1 true}
!36 = distinct !{null, null, null, i1 false, i1 true}
!37 = distinct !{null, null, null, i1 false, i1 true}
!38 = !{i32 1, !"Objective-C Version", i32 2}
!39 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!40 = !{i32 1, !"Objective-C Image Info Section", !"__DATA,__objc_imageinfo,regular,no_dead_strip"}
!41 = !{i32 1, !"Objective-C Garbage Collection", i8 0}
!42 = !{i32 1, !"Objective-C Class Properties", i32 64}
!43 = !{i32 7, !"Dwarf Version", i32 4}
!44 = !{i32 2, !"Debug Info Version", i32 3}
!45 = !{i32 1, !"wchar_size", i32 4}
!46 = !{i32 1, !"branch-target-enforcement", i32 0}
!47 = !{i32 1, !"sign-return-address", i32 0}
!48 = !{i32 1, !"sign-return-address-all", i32 0}
!49 = !{i32 1, !"sign-return-address-with-bkey", i32 0}
!50 = !{i32 7, !"PIC Level", i32 2}
!51 = !{i32 1, !"Swift Version", i32 7}
!52 = !{i32 1, !"Swift ABI Version", i32 7}
!53 = !{i32 1, !"Swift Major Version", i8 5}
!54 = !{i32 1, !"Swift Minor Version", i8 5}
!55 = !{!"-lswiftSwiftOnoneSupport"}
!56 = !{!"-lswiftCore"}
!57 = !{!"-lswift_Concurrency"}
!58 = !{!"-lobjc"}
!59 = distinct !DISubprogram(name: "fibonacci", linkageName: "$s3fib9FibonacciC9fibonacciyS2iYF", scope: !60, file: !1, line: 4, type: !61, scopeLine: 9, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!60 = !DICompositeType(tag: DW_TAG_structure_type, name: "Fibonacci", scope: !5, file: !1, size: 64, elements: !2, runtimeLang: DW_LANG_Swift, identifier: "$s3fib9FibonacciCD")
!61 = !DISubroutineType(types: !62)
!62 = !{!63, !63, !60}
!63 = !DICompositeType(tag: DW_TAG_structure_type, name: "Int", scope: !7, file: !64, size: 64, elements: !2, runtimeLang: DW_LANG_Swift, identifier: "$sSiD")
!64 = !DIFile(filename: "_build/Ninja-RelWithDebInfoAssert/swift-macosx-x86_64/lib/swift/macosx/Swift.swiftmodule/arm64-apple-macos.swiftmodule", directory: "/Volumes/Data/swift")
!65 = !DILocalVariable(name: "res", scope: !59, file: !1, line: 11, type: !66)
!66 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !63)
!67 = !DILocation(line: 11, column: 14, scope: !59)
!68 = !DILocalVariable(name: "n_2", scope: !59, file: !1, line: 9, type: !66)
!69 = !DILocation(line: 9, column: 14, scope: !59)
!70 = !DILocalVariable(name: "n_1", scope: !59, file: !1, line: 8, type: !66)
!71 = !DILocation(line: 8, column: 14, scope: !59)
!72 = !DILocalVariable(name: "self", arg: 2, scope: !59, file: !1, line: 4, type: !73, flags: DIFlagArtificial)
!73 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !60)
!74 = !DILocation(line: 4, column: 10, scope: !59)
!75 = !DILocalVariable(name: "n", arg: 1, scope: !59, file: !1, line: 4, type: !66)
!76 = !DILocation(line: 4, column: 20, scope: !59)
!77 = !DILocation(line: 0, scope: !78, inlinedAt: !81)
!78 = distinct !DISubprogram(linkageName: "__swift_async_resume_project_context", scope: !5, file: !79, type: !80, flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!79 = !DIFile(filename: "<compiler-generated>", directory: "")
!80 = !DISubroutineType(types: null)
!81 = distinct !DILocation(line: 9, column: 26, scope: !59)
!82 = !DILocation(line: 9, column: 26, scope: !59)
!83 = !DILocation(line: 0, scope: !78, inlinedAt: !84)
!84 = distinct !DILocation(line: 9, column: 26, scope: !59)
!85 = !DILocation(line: 0, scope: !86, inlinedAt: !87)
!86 = distinct !DISubprogram(linkageName: "__swift_suspend_point", scope: !5, file: !79, type: !80, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!87 = distinct !DILocation(line: 9, column: 26, scope: !59)
