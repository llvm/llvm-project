; RUN: rm -rf %t && mkdir -p %t
; RUN: llc --filetype=obj --mccas-verify --cas-backend --cas=%t/cas %s -o %t/dwarf-5.o 
; RUN: llvm-dwarfdump %t/dwarf-5.o | FileCheck %s
; CHECK: .debug_info contents:
; CHECK-NEXT: 0x{{[0-9a-f]+}}: Compile Unit: length = 0x{{[0-9a-f]+}}, format = DWARF32, version = 0x0005

source_filename = "/Users/shubham/Development/Delta/alternate/CommandLine.cpp"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx14.0.0"

%"class.llvm::StringRef" = type { ptr, i64 }
%"class.llvm::function_ref" = type { ptr, i64 }
%"class.llvm::function_ref.1" = type { ptr, i64 }
%"class.llvm::SmallString" = type { %"class.llvm::SmallVector" }
%"class.llvm::SmallVector" = type { %"class.llvm::SmallVectorImpl.2" }
%"class.llvm::SmallVectorImpl.2" = type { %"class.llvm::SmallVectorTemplateBase.3" }
%"class.llvm::SmallVectorTemplateBase.3" = type { %"class.llvm::SmallVectorTemplateCommon.4" }
%"class.llvm::SmallVectorTemplateCommon.4" = type { %"class.llvm::SmallVectorBase" }
%"class.llvm::SmallVectorBase" = type { ptr, i64, i64 }
define void @_ZN4llvm2cl26TokenizeWindowsCommandLineENS_9StringRefERNS_11StringSaverERNS_15SmallVectorImplIPKcEEb([2 x i64] %0, ptr noundef nonnull align 1 dereferenceable(1) %1, ptr noundef nonnull align 8 dereferenceable(24) %2, i1 noundef zeroext %3) #0 !dbg !114 {
  ret void, !dbg !195
}
define internal void @_ZL30tokenizeWindowsCommandLineImplN4llvm9StringRefERNS_11StringSaverENS_12function_refIFvS0_EEEbNS3_IFvvEEEb([2 x i64] %0, ptr noundef nonnull align 1 dereferenceable(1) %1, [2 x i64] %2, i1 noundef zeroext %3, [2 x i64] %4, i1 noundef zeroext %5) #0 !dbg !12 {
  %7 = alloca %"class.llvm::StringRef", align 8
  %8 = alloca %"class.llvm::function_ref", align 8
  %9 = alloca %"class.llvm::function_ref.1", align 8
  %10 = alloca ptr, align 8
  %11 = alloca i8, align 1
  %12 = alloca i8, align 1
  %13 = alloca %"class.llvm::SmallString", align 8
  %14 = alloca i32, align 4
  %15 = alloca i64, align 8
  %16 = alloca i64, align 8
  %17 = alloca %"class.llvm::StringRef", align 8
  %18 = zext i1 %3 to i8
  %19 = zext i1 %5 to i8
  %20 = call noundef ptr @_ZN4llvm11SmallStringILj128EEC1Ev(ptr noundef nonnull align 8 dereferenceable(24) %13), !dbg !256
  %21 = call noundef i64 @_ZNK4llvm9StringRef4sizeEv(ptr noundef nonnull align 8 dereferenceable(16) %7), !dbg !264
  br label %22, !dbg !265
  br label %22, !dbg !283, !llvm.loop !284
}
define internal noundef ptr @"_ZN4llvm12function_refIFvvEEC1IRZNS_2cl26TokenizeWindowsCommandLineENS_9StringRefERNS_11StringSaverERNS_15SmallVectorImplIPKcEEbE3$_1EEOT_PNSt3__19enable_ifIXooL_ZNSH_17integral_constantIbLb1EE5valueEEsr3std14is_convertibleIDTclclsr3stdE7declvalISF_EEEEvEE5valueEvE4typeE"(ptr noundef nonnull returned align 8 dereferenceable(16) %0, ptr noundef nonnull align 1 dereferenceable(1) %1, ptr noundef %2) unnamed_addr #3 align 2 !dbg !312 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = load ptr, ptr %4, align 8
  ret ptr %7, !dbg !330
}
define linkonce_odr noundef ptr @_ZN4llvm11SmallStringILj128EEC1Ev(ptr noundef nonnull returned align 8 dereferenceable(24) %0) unnamed_addr #3 align 2 !dbg !331 {
  %2 = alloca ptr, align 8
  %3 = load ptr, ptr %2, align 8
  ret ptr %3, !dbg !339
}
define linkonce_odr noundef i64 @_ZNK4llvm9StringRef4sizeEv(ptr noundef nonnull align 8 dereferenceable(16) %0) #0 align 2 !dbg !340 {
  %2 = alloca ptr, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds %"class.llvm::StringRef", ptr %3, i32 0, i32 1, !dbg !344
  %5 = load i64, ptr %4, align 8, !dbg !344
  ret i64 %5, !dbg !345
}
declare void @llvm.trap() #4
define linkonce_odr noundef ptr @_ZNSt3__122__uninitialized_fill_nIcPcmcEET0_S2_T1_RKT2_(ptr noundef %0, i64 noundef %1, ptr noundef nonnull align 1 dereferenceable(1) %2) #0 !dbg !469 {
  call void @llvm.trap(), !dbg !489
  unreachable, !dbg !491
}
define linkonce_odr noundef ptr @_ZN4llvm25SmallVectorTemplateCommonIcvE5beginEv(ptr noundef nonnull align 8 dereferenceable(24) %0) #0 align 2 !dbg !492 {
  %2 = alloca ptr, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds %"class.llvm::SmallVectorBase", ptr %3, i32 0, i32 0, !dbg !495
  %5 = load ptr, ptr %4, align 8, !dbg !495
  ret ptr %5, !dbg !496
}
!llvm.module.flags = !{!1, !2, !6}
!llvm.dbg.cu = !{!7}
!1 = !{i32 7, !"Dwarf Version", i32 5}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 7, !"frame-pointer", i32 1}
!7 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !8, emissionKind: FullDebug, sdk: "MacOSX14.0.sdk")
!8 = !DIFile(filename: "/Users/shubham/Development/Delta/alternate/CommandLine.cpp", directory: "/Users/shubham/Development/Delta/alternate", checksumkind: CSK_MD5, checksum: "ed7ae158f20f7914bc5fb843291e80da")
!12 = distinct !DISubprogram(name: "tokenizeWindowsCommandLineImpl", type: !13, unit: !7, retainedNodes: !36)
!13 = !DISubroutineType(types: !14)
!14 = !{}
!32 = !DISubroutineType(types: !33)
!33 = !{}
!36 = !{}
!101 = !DISubroutineType(types: !102)
!102 = !{}
!114 = distinct !DISubprogram(name: "TokenizeWindowsCommandLine", type: !116, unit: !7, retainedNodes: !36)
!116 = !DISubroutineType(types: !117)
!117 = !{}
!195 = !DILocation(line: 428, scope: !114)
!256 = !DILocation(line: 410, scope: !12)
!260 = distinct !DILexicalBlock(scope: !12, line: 412, column: 3)
!264 = !DILocation(line: 412, scope: !260)
!265 = !DILocation(line: 412, scope: !260)
!267 = distinct !DILexicalBlock(scope: !260, line: 412, column: 20)
!283 = !DILocation(line: 412, scope: !267)
!284 = distinct !{}
!312 = distinct !DISubprogram(name: "function_ref<(lambda at /Users/shubham/Development/Delta/alternate/CommandLine.cpp:424:16) &>", type: !313, unit: !7, retainedNodes: !36)
!313 = !DISubroutineType(types: !314)
!314 = !{}
!330 = !DILocation(line: 348, scope: !312)
!331 = distinct !DISubprogram(name: "SmallString", type: !332, unit: !7, retainedNodes: !36)
!332 = !DISubroutineType(types: !333)
!333 = !{}
!339 = !DILocation(line: 399, scope: !331)
!340 = distinct !DISubprogram(name: "size", type: !32, unit: !7, retainedNodes: !36)
!344 = !DILocation(line: 372, scope: !340)
!345 = !DILocation(line: 372, scope: !340)
!444 = !DISubroutineType(types: !445)
!445 = !{}
!462 = distinct !DISubprogram(name: "end", scope: !462)
!469 = distinct !DISubprogram(name: "__uninitialized_fill_n<char, char>", type: !444, unit: !7, retainedNodes: !36)
!483 = distinct !DISubprogram(name: "__voidify<char>", unit: !7, retainedNodes: !36)
!488 = distinct !DILocation(line: 214, scope: !469)
!489 = !DILocation(line: 137, scope: !483, inlinedAt: !488)
!491 = !DILocation(line: 214, scope: !469)
!492 = distinct !DISubprogram(name: "begin", type: !101, unit: !7, retainedNodes: !36)
!495 = !DILocation(line: 290, scope: !492)
!496 = !DILocation(line: 290, scope: !492)
