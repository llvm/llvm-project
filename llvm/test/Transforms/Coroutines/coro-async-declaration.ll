; RUN: opt < %s -passes='default<O2>' -S | FileCheck %s

; The code is from https://github.com/apple/llvm-project/blob/5c3acb099acec3f644d810ce67fb8b7076f2621a/lldb/test/API/lang/swift/async/stepping/step-in/task-switch/main.swift.
; This is a DISubprogram with declaration.

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx12.0.0"

%swift.async_func_pointer = type <{ i32, i32 }>
%swift.protocol_conformance_descriptor = type { i32, i32, i32, i32 }
%swift.type = type { i64 }
%swift.full_existential_type = type { ptr, %swift.type }
%swift.full_boxmetadata = type { ptr, ptr, %swift.type, i32, ptr }
%swift.metadata_response = type { ptr, i64 }
%Any = type { [24 x i8], ptr }
%TSi = type <{ i64 }>
%swift.async_task_and_context = type { ptr, ptr }

@"$s4main5entryOAAyyYaFZTu" = hidden global %swift.async_func_pointer <{ i32 trunc (i64 sub (i64 ptrtoint (ptr @"$s4main5entryOAAyyYaFZ" to i64), i64 ptrtoint (ptr @"$s4main5entryOAAyyYaFZTu" to i64)) to i32), i32 16 }>, align 8
@"$s4main1fSiyYaFTu" = hidden global %swift.async_func_pointer <{ i32 trunc (i64 sub (i64 ptrtoint (ptr @"$s4main1fSiyYaF" to i64), i64 ptrtoint (ptr @"$s4main1fSiyYaFTu" to i64)) to i32), i32 16 }>, align 8
@"$sS2cMScAsWL" = linkonce_odr hidden global ptr null, align 8
@"$sScMScAsMc" = external global %swift.protocol_conformance_descriptor, align 4
@"$sSiN" = external global %swift.type, align 8
@"$sypN" = external global %swift.full_existential_type
@"$sytN" = external global %swift.full_existential_type
@"symbolic IetH_" = linkonce_odr hidden constant <{ [5 x i8], i8 }> <{ [5 x i8] c"IetH_", i8 0 }>, section "__TEXT,__swift5_typeref, regular", no_sanitize_address, align 2
@"\01l__swift5_reflection_descriptor" = private constant { i32, i32, i32, i32 } { i32 1, i32 0, i32 0, i32 trunc (i64 sub (i64 ptrtoint (ptr @"symbolic IetH_" to i64), i64 ptrtoint (ptr getelementptr inbounds ({ i32, i32, i32, i32 }, ptr @"\01l__swift5_reflection_descriptor", i32 0, i32 3) to i64)) to i32) }, section "__TEXT,__swift5_capture, regular", no_sanitize_address, align 4
@metadata = private constant %swift.full_boxmetadata { ptr @objectdestroy, ptr null, %swift.type { i64 1024 }, i32 16, ptr @"\01l__swift5_reflection_descriptor" }, align 8
@"$sytWV" = external global ptr, align 8
@.str.4.main = private constant [5 x i8] c"main\00"
@"$s4mainMXM" = linkonce_odr hidden constant <{ i32, i32, i32 }> <{ i32 0, i32 0, i32 trunc (i64 sub (i64 ptrtoint (ptr @.str.4.main to i64), i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32, i32 }>, ptr @"$s4mainMXM", i32 0, i32 2) to i64)) to i32) }>, section "__TEXT,__constg_swiftt", align 4
@.str.5.entry = private constant [6 x i8] c"entry\00"
@".str.1.\0A" = private unnamed_addr constant [2 x i8] c"\0A\00"
@".str.1. " = private unnamed_addr constant [2 x i8] c" \00"
@".str.25.Swift/BridgeStorage.swift" = private unnamed_addr constant [26 x i8] c"Swift/BridgeStorage.swift\00"
@.str.0. = private unnamed_addr constant [1 x i8] zeroinitializer
@".str.11.Fatal error" = private unnamed_addr constant [12 x i8] c"Fatal error\00"
@__swift_reflection_version = linkonce_odr hidden constant i16 3

define hidden swifttailcc void @"$s4main5entryOAAyyYaFZ"(ptr swiftasync %0) #0 !dbg !38 {
entry:
  %1 = alloca ptr, align 8
  %x.debug = alloca i64, align 8
  %2 = call token @llvm.coro.id.async(i32 16, i32 16, i32 0, ptr @"$s4main5entryOAAyyYaFZTu")
  %3 = call ptr @llvm.coro.begin(token %2, ptr null)
  store ptr %0, ptr %1, align 8
  call void @llvm.memset.p0.i64(ptr align 8 %x.debug, i8 0, i64 8, i1 false)
  %4 = call swiftcc %swift.metadata_response @"$sScMMa"(i64 0) #13
  %5 = extractvalue %swift.metadata_response %4, 0
  %6 = call swiftcc ptr @"$sScM6sharedScMvgZ"(ptr swiftself %5)
  %7 = load i32, ptr getelementptr inbounds (%swift.async_func_pointer, ptr @"$s4main1fSiyYaFTu", i32 0, i32 1), align 8
  %8 = zext i32 %7 to i64
  %9 = call swiftcc ptr @swift_task_alloc(i64 %8) #3
  call void @llvm.lifetime.start.p0(i64 -1, ptr %9)
  %10 = load ptr, ptr %1, align 8
  %11 = getelementptr inbounds <{ ptr, ptr }>, ptr %9, i32 0, i32 0
  store ptr %10, ptr %11, align 8
  %12 = call ptr @llvm.coro.async.resume()
  %13 = getelementptr inbounds <{ ptr, ptr }>, ptr %9, i32 0, i32 1
  store ptr %12, ptr %13, align 8
  %14 = call { ptr, i64 } (i32, ptr, ptr, ...) @llvm.coro.suspend.async.sl_p0i64s(i32 0, ptr %12, ptr @__swift_async_resume_project_context, ptr @"$s4main5entryOAAyyYaFZ.0", ptr @"$s4main1fSiyYaF", ptr %9)
  %15 = extractvalue { ptr, i64 } %14, 0
  %16 = call ptr @__swift_async_resume_project_context(ptr %15)
  store ptr %16, ptr %1, align 8
  %17 = extractvalue { ptr, i64 } %14, 1
  call swiftcc void @swift_task_dealloc(ptr %9) #3
  call void @llvm.lifetime.end.p0(i64 -1, ptr %9)
  %18 = call ptr @"$sS2cMScAsWl"() #13
  %19 = call swiftcc { i64, i64 } @"$sScA15unownedExecutorScevgTj"(ptr swiftself %6, ptr %5, ptr %18)
  %20 = extractvalue { i64, i64 } %19, 0
  %21 = extractvalue { i64, i64 } %19, 1
  %22 = call ptr @llvm.coro.async.resume()
  %23 = load ptr, ptr %1, align 8
  %24 = load ptr, ptr %1, align 8
  %25 = call { ptr } (i32, ptr, ptr, ...) @llvm.coro.suspend.async.sl_p0s(i32 0, ptr %22, ptr @__swift_async_resume_get_context, ptr @"$s4main5entryOAAyyYaFZ.1", ptr %22, i64 %20, i64 %21, ptr %24)
  %26 = extractvalue { ptr } %25, 0
  %27 = call ptr @__swift_async_resume_get_context(ptr %26)
  store ptr %27, ptr %1, align 8
  store i64 %17, ptr %x.debug, align 8
  call void asm sideeffect "", "r"(ptr %x.debug)
  %28 = call swiftcc { ptr, ptr } @"$ss27_allocateUninitializedArrayySayxG_BptBwlFyp_Tg5"(i64 1)
  %29 = extractvalue { ptr, ptr } %28, 0
  %30 = extractvalue { ptr, ptr } %28, 1
  %31 = getelementptr inbounds %Any, ptr %30, i32 0, i32 1
  store ptr @"$sSiN", ptr %31, align 8
  %32 = getelementptr inbounds %Any, ptr %30, i32 0, i32 0
  %33 = getelementptr inbounds %Any, ptr %30, i32 0, i32 0
  %._value = getelementptr inbounds %TSi, ptr %33, i32 0, i32 0
  store i64 %17, ptr %._value, align 8
  %34 = call swiftcc ptr @"$ss27_finalizeUninitializedArrayySayxGABnlF"(ptr %29, ptr getelementptr inbounds (%swift.full_existential_type, ptr @"$sypN", i32 0, i32 1))
  %35 = call swiftcc { i64, ptr } @"$ss5print_9separator10terminatoryypd_S2StFfA0_"()
  %36 = extractvalue { i64, ptr } %35, 0
  %37 = extractvalue { i64, ptr } %35, 1
  %38 = call swiftcc { i64, ptr } @"$ss5print_9separator10terminatoryypd_S2StFfA1_"()
  %39 = extractvalue { i64, ptr } %38, 0
  %40 = extractvalue { i64, ptr } %38, 1
  call swiftcc void @"$ss5print_9separator10terminatoryypd_S2StF"(ptr %34, i64 %36, ptr %37, i64 %39, ptr %40)
  call void @swift_bridgeObjectRelease(ptr %40) #1
  call void @swift_bridgeObjectRelease(ptr %37) #1
  call void @swift_bridgeObjectRelease(ptr %34) #1
  call void @swift_release(ptr %6) #1
  call void asm sideeffect "", "r"(ptr %x.debug)
  %41 = load ptr, ptr %1, align 8
  %42 = getelementptr inbounds <{ ptr, ptr }>, ptr %41, i32 0, i32 1
  %43 = load ptr, ptr %42, align 8
  %44 = load ptr, ptr %1, align 8
  %45 = call i1 (ptr, i1, ...) @llvm.coro.end.async(ptr %3, i1 false, ptr @"$s4main5entryOAAyyYaFZ.0.1", ptr %43, ptr %44)
  unreachable
}

; Function Attrs: nounwind
declare token @llvm.coro.id.async(i32, i32, i32, ptr) #1

; Function Attrs: cold noreturn nounwind memory(inaccessiblemem: write)
declare void @llvm.trap() #2

; Function Attrs: nounwind
declare ptr @llvm.coro.begin(token, ptr writeonly) #1

declare swiftcc %swift.metadata_response @"$sScMMa"(i64) #0

declare swiftcc ptr @"$sScM6sharedScMvgZ"(ptr swiftself) #0

declare swifttailcc void @"$s4main1fSiyYaF"(ptr swiftasync) #0

; Function Attrs: nounwind memory(argmem: readwrite)
declare swiftcc ptr @swift_task_alloc(i64) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #4

; Function Attrs: nomerge nounwind
declare ptr @llvm.coro.async.resume() #5

; Function Attrs: alwaysinline nounwind
define linkonce_odr hidden ptr @__swift_async_resume_project_context(ptr %0) #6 {
entry:
  %1 = load ptr, ptr %0, align 8
  %2 = call ptr @llvm.swift.async.context.addr()
  store ptr %1, ptr %2, align 8
  ret ptr %1
}

; Function Attrs: nounwind
declare ptr @llvm.swift.async.context.addr() #1

; Function Attrs: nounwind
define internal swifttailcc void @"$s4main5entryOAAyyYaFZ.0"(ptr %0, ptr %1) #1 {
entry:
  musttail call swifttailcc void %0(ptr swiftasync %1)
  ret void
}

; Function Attrs: nomerge nounwind
declare { ptr, i64 } @llvm.coro.suspend.async.sl_p0i64s(i32, ptr, ptr, ...) #5

; Function Attrs: nounwind memory(argmem: readwrite)
declare swiftcc void @swift_task_dealloc(ptr) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #4

declare swiftcc { i64, i64 } @"$sScA15unownedExecutorScevgTj"(ptr swiftself, ptr, ptr) #0

; Function Attrs: noinline nounwind memory(none)
declare ptr @"$sS2cMScAsWl"() #7

; Function Attrs: nounwind memory(read)
declare ptr @swift_getWitnessTable(ptr, ptr, ptr) #8

; Function Attrs: nounwind
define linkonce_odr hidden ptr @__swift_async_resume_get_context(ptr %0) #9 {
entry:
  ret ptr %0
}

; Function Attrs: nounwind
define internal swifttailcc void @"$s4main5entryOAAyyYaFZ.1"(ptr %0, i64 %1, i64 %2, ptr %3) #1 {
entry:
  musttail call swifttailcc void @swift_task_switch(ptr swiftasync %3, ptr %0, i64 %1, i64 %2) #1
  ret void
}

; Function Attrs: nounwind
declare swifttailcc void @swift_task_switch(ptr, ptr, i64, i64) #1

; Function Attrs: nomerge nounwind
declare { ptr } @llvm.coro.suspend.async.sl_p0s(i32, ptr, ptr, ...) #5

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #10

declare swiftcc { ptr, ptr } @"$ss27_allocateUninitializedArrayySayxG_BptBwlFyp_Tg5"(i64) #0

declare swiftcc ptr @"$ss27_finalizeUninitializedArrayySayxGABnlF"(ptr, ptr) #0

declare swiftcc { i64, ptr } @"$ss5print_9separator10terminatoryypd_S2StFfA0_"() #0

declare swiftcc { i64, ptr } @"$ss5print_9separator10terminatoryypd_S2StFfA1_"() #0

declare swiftcc void @"$ss5print_9separator10terminatoryypd_S2StF"(ptr, i64, ptr, i64, ptr) #0

; Function Attrs: nounwind
declare void @swift_bridgeObjectRelease(ptr) #1

; Function Attrs: nounwind
declare void @swift_release(ptr) #1

; Function Attrs: nounwind
define internal swifttailcc void @"$s4main5entryOAAyyYaFZ.0.1"(ptr %0, ptr %1) #1 {
entry:
  musttail call swifttailcc void %0(ptr swiftasync %1)
  ret void
}

; Function Attrs: nounwind
declare i1 @llvm.coro.end.async(ptr, i1, ...) #1

; Function Attrs: nounwind
declare swifttailcc void @"$s4main1fSiyYaF.0"(ptr, ptr, i64) #1

; Function Attrs: noreturn
declare void @exit(i32 noundef) #11

declare swiftcc void @objectdestroy(ptr swiftself) #0

; Function Attrs: nounwind
declare void @swift_deallocObject(ptr, i64, i64) #1

; Function Attrs: nounwind
declare ptr @swift_allocObject(ptr, i64, i64) #1

; Function Attrs: nomerge nounwind
declare { ptr, ptr } @llvm.coro.suspend.async.sl_p0p0s(i32, ptr, ptr, ...) #5

; Function Attrs: nounwind memory(argmem: readwrite)
declare swiftcc %swift.async_task_and_context @swift_task_create(i64, i64, ptr, ptr, ptr) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare i1 @llvm.expect.i1(i1, i1) #12

attributes #0 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind }
attributes #2 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #3 = { nounwind memory(argmem: readwrite) }
attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { nomerge nounwind }
attributes #6 = { alwaysinline nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" }
attributes #7 = { noinline nounwind memory(none) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" }
attributes #8 = { nounwind memory(read) }
attributes #9 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" }
attributes #10 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #11 = { noreturn "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" }
attributes #12 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #13 = { nounwind memory(none) }

!llvm.dbg.cu = !{!0}
!swift.module.flags = !{!15}
!llvm.module.flags = !{!16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30}
!llvm.linker.options = !{!31, !32, !33, !34, !35, !36, !37}

!0 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !1, producer: "Apple Swift version 5.9-dev (LLVM 79f8de4f4f8ad05, Swift 3f522148c284926)", isOptimized: false, runtimeVersion: 5, emissionKind: FullDebug, imports: !2)
!1 = !DIFile(filename: "/Users/dianqk/swift-project/llvm-project/lldb/test/API/lang/swift/async/stepping/step-in/task-switch/main.swift", directory: "/Users/dianqk/swift-project")
!2 = !{!3, !5, !7, !9, !11, !13}
!3 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !1, entity: !4, file: !1)
!4 = !DIModule(scope: null, name: "main", includePath: "/Users/dianqk/swift-project/llvm-project/lldb/test/API/lang/swift/async/stepping/step-in/task-switch")
!5 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !1, entity: !6, file: !1)
!6 = !DIModule(scope: null, name: "Swift", includePath: "/Users/dianqk/swift-project/build/buildbot_incremental/swift-macosx-x86_64/lib/swift/macosx/Swift.swiftmodule/x86_64-apple-macos.swiftmodule")
!7 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !1, entity: !8, file: !1)
!8 = !DIModule(scope: null, name: "_StringProcessing", includePath: "/Users/dianqk/swift-project/build/buildbot_incremental/swift-macosx-x86_64/lib/swift/macosx/_StringProcessing.swiftmodule/x86_64-apple-macos.swiftmodule")
!9 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !1, entity: !10, file: !1)
!10 = !DIModule(scope: null, name: "_SwiftConcurrencyShims", includePath: "/Users/dianqk/swift-project/build/buildbot_incremental/swift-macosx-x86_64/lib/swift/shims")
!11 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !1, entity: !12, file: !1)
!12 = !DIModule(scope: null, name: "_Concurrency", includePath: "/Users/dianqk/swift-project/build/buildbot_incremental/swift-macosx-x86_64/lib/swift/macosx/_Concurrency.swiftmodule/x86_64-apple-macos.swiftmodule")
!13 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !1, entity: !14, file: !1)
!14 = !DIModule(scope: null, name: "SwiftOnoneSupport", includePath: "/Users/dianqk/swift-project/build/buildbot_incremental/swift-macosx-x86_64/lib/swift/macosx/SwiftOnoneSupport.swiftmodule/x86_64-apple-macos.swiftmodule")
!15 = !{!"standard-library", i1 false}
!16 = !{i32 1, !"Objective-C Version", i32 2}
!17 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!18 = !{i32 1, !"Objective-C Image Info Section", !"__DATA,__objc_imageinfo,regular,no_dead_strip"}
!19 = !{i32 1, !"Objective-C Garbage Collection", i8 0}
!20 = !{i32 1, !"Objective-C Class Properties", i32 64}
!21 = !{i32 7, !"Dwarf Version", i32 4}
!22 = !{i32 2, !"Debug Info Version", i32 3}
!23 = !{i32 1, !"wchar_size", i32 4}
!24 = !{i32 8, !"PIC Level", i32 2}
!25 = !{i32 7, !"uwtable", i32 2}
!26 = !{i32 7, !"frame-pointer", i32 2}
!27 = !{i32 1, !"Swift Version", i32 7}
!28 = !{i32 1, !"Swift ABI Version", i32 7}
!29 = !{i32 1, !"Swift Major Version", i8 5}
!30 = !{i32 1, !"Swift Minor Version", i8 9}
!31 = !{!"-lswiftSwiftOnoneSupport"}
!32 = !{!"-lswiftCore"}
!33 = !{!"-lswift_Concurrency"}
!34 = !{!"-lswift_StringProcessing"}
!35 = !{!"-lobjc"}
!36 = !{!"-lswiftCompatibility56"}
!37 = !{!"-lswiftCompatibilityPacks"}
!38 = distinct !DISubprogram(name: "main", linkageName: "$s4main5entryOAAyyYaFZ", scope: !39, file: !1, line: 2, type: !40, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, declaration: !45, retainedNodes: !46)
!39 = !DICompositeType(tag: DW_TAG_structure_type, name: "$s4main5entryOD", scope: !4, flags: DIFlagFwdDecl, runtimeLang: DW_LANG_Swift)
!40 = !DISubroutineType(types: !41)
!41 = !{!42, !43}
!42 = !DICompositeType(tag: DW_TAG_structure_type, name: "$sytD", flags: DIFlagFwdDecl, runtimeLang: DW_LANG_Swift)
!43 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "$s4main5entryOXMtD", file: !44, size: 8, flags: DIFlagArtificial, runtimeLang: DW_LANG_Swift, identifier: "$s4main5entryOXMtD")
!44 = !DIFile(filename: "<compiler-generated>", directory: "")
; CHECK-DAG: ![[DECL:[0-9]+]] = !DISubprogram({{.*}}, linkageName: "$s4main5entryOAAyyYaFZ"
; CHECK-DAG: ![[DECL_Q0:[0-9]+]] = !DISubprogram({{.*}}, linkageName: "$s4main5entryOAAyyYaFZTQ0_"
; CHECK-DAG: ![[DECL_Y1:[0-9]+]] = !DISubprogram({{.*}}, linkageName: "$s4main5entryOAAyyYaFZTY1_"
; CHECK-DAG: distinct !DISubprogram({{.*}}, linkageName: "$s4main5entryOAAyyYaFZ"{{.*}}, declaration: ![[DECL]]
; CHECK-DAG: distinct !DISubprogram({{.*}}, linkageName: "$s4main5entryOAAyyYaFZTQ0_"{{.*}}, declaration: ![[DECL_Q0]]
; CHECK-DAG: distinct !DISubprogram({{.*}}, linkageName: "$s4main5entryOAAyyYaFZTY1_"{{.*}}, declaration: ![[DECL_Y1]]
!45 = !DISubprogram(name: "main", linkageName: "$s4main5entryOAAyyYaFZ", scope: !39, file: !1, line: 2, type: !40, scopeLine: 2, spFlags: 0)
!46 = !{!47, !49}
!47 = !DILocalVariable(name: "self", arg: 1, scope: !38, file: !1, line: 2, type: !48, flags: DIFlagArtificial)
!48 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !43)
!49 = !DILocalVariable(name: "x", scope: !50, file: !1, line: 3, type: !51)
!50 = distinct !DILexicalBlock(scope: !38, file: !1, line: 3, column: 9)
!51 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !52)
!52 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Int", scope: !6, file: !53, size: 64, elements: !54, runtimeLang: DW_LANG_Swift, identifier: "$sSiD")
!53 = !DIFile(filename: "build/buildbot_incremental/swift-macosx-x86_64/lib/swift/macosx/Swift.swiftmodule/x86_64-apple-macos.swiftmodule", directory: "/Users/dianqk/swift-project")
!54 = !{}
