; RUN: llvm-as -disable-output %s
%"class.llvm::StringRef" = type { ptr, i64 }
define internal void @_ZL30tokenizeWindowsCommandLineImplN4llvm9StringRefERNS_11StringSaverENS_12function_refIFvS0_EEEbNS3_IFvvEEEb() !dbg !12 {
  %7 = alloca %"class.llvm::StringRef", align 8
  %21 = call noundef i64 @_ZNK4llvm9StringRef4sizeEv(ptr noundef nonnull align 8 dereferenceable(16) %7)
  br label %22
  br label %22, !llvm.loop !284 ; This instruction has loop metadata but no operands and should not result in a segmentation fault in the verifier.
}
define linkonce_odr noundef i64 @_ZNK4llvm9StringRef4sizeEv() align 2 !dbg !340 {
  %2 = alloca ptr, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds %"class.llvm::StringRef", ptr %3
  %5 = load i64, ptr %4
  ret i64 %5
}
!llvm.module.flags = !{!2, !6}
!llvm.dbg.cu = !{!7}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 7, !"frame-pointer", i32 1}
!7 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !8, sdk: "MacOSX14.0.sdk")
!8 = !DIFile(filename: "file.cpp", directory: "/Users/Dev", checksumkind: CSK_MD5, checksum: "ed7ae158f20f7914bc5fb843291e80da")
!12 = distinct !DISubprogram(unit: !7, retainedNodes: !36)
!36 = !{}
!284 = distinct !{}
!340 = distinct !DISubprogram(unit: !7, retainedNodes: !36)
