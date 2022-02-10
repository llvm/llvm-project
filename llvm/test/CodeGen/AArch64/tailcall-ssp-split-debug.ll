; RUN: llc -mtriple=arm64-apple-ios %s -o - | FileCheck %s

define swifttailcc void @foo(i8* %call) ssp {
; CHECK-LABEL: foo:
  %var = alloca [28 x i8], align 16
  br i1 undef, label %if.then, label %if.end

if.then:
  ret void

if.end:
  ; CHECK: mov x[[NULL:[0-9]+]], xzr
  ; CHECK: ldr [[FPTR:x[0-9]+]], [x[[NULL]]]
  ; CHECK: br [[FPTR]]
  call void @llvm.dbg.value(metadata i8* %call, metadata !19, metadata !DIExpression()), !dbg !21
  %fptr = load void (i8*)*, void (i8*)** null, align 8
  musttail call swifttailcc void %fptr(i8* null)
  ret void
}

declare i8* @pthread_getspecific()

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

!llvm.module.flags = !{!2}
!llvm.dbg.cu = !{!11}

!2 = !{i32 2, !"Debug Info Version", i32 3}
!11 = distinct !DICompileUnit(language: DW_LANG_C99, file: !12, producer: "Apple clang version 13.1.6 (clang-1316.0.17.4)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !13, splitDebugInlining: false, nameTableKind: None, sysroot: "/Library/Developer/CommandLineTools/SDKs/MacOSX12.3.sdk", sdk: "MacOSX12.3.sdk")
!12 = !DIFile(filename: "tmp.c", directory: "/Users/tim/llvm-internal/llvm-project/build")
!13 = !{}
!14 = !{!"Apple clang version 13.1.6 (clang-1316.0.17.4)"}
!15 = distinct !DISubprogram(name: "bar", scope: !12, file: !12, line: 3, type: !16, scopeLine: 3, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !11, retainedNodes: !18)
!16 = !DISubroutineType(types: !17)
!17 = !{null}
!18 = !{!19}
!19 = !DILocalVariable(name: "var", scope: !15, file: !12, line: 4, type: !20)
!20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!21 = !DILocation(line: 0, scope: !15)
