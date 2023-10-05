; This checks that .debug_names is not generated with split-dwarf + -fdebug-type-sections.

; RUN: llc -mtriple=x86_64 -generate-type-units -dwarf-version=5 -filetype=obj -split-dwarf-file=mainTypes.dwo --split-dwarf-output=mainTypes.dwo %s -o %t
; RUN: llvm-readelf --sections %t | FileCheck %s

; CHECK-NOT: .debug_names

; ModuleID = 'main.cpp'
source_filename = "main.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Foo = type { ptr }

; Function Attrs: mustprogress noinline norecurse nounwind optnone uwtable
define dso_local noundef i32 @main() #0 !dbg !10 {
entry:
  %retval = alloca i32, align 4
  %f = alloca %struct.Foo, align 8
  store i32 0, ptr %retval, align 4
  call void @llvm.dbg.declare(metadata ptr %f, metadata !15, metadata !DIExpression()), !dbg !21
  ret i32 0, !dbg !22
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { mustprogress noinline norecurse nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 18.0.0 (ssh://git.vip.facebook.com/data/gitrepos/osmeta/external/llvm-project 680deb27e25976d9b74ad7b48ec5cb96be0e44b6)", isOptimized: false, runtimeVersion: 0, splitDebugFilename: "main.dwo", emissionKind: FullDebug, splitDebugInlining: false)
!1 = !DIFile(filename: "main.cpp", directory: "/home/ayermolo/local/tasks/T138552329/typeSmallSplit", checksumkind: CSK_MD5, checksum: "e5b402e9dbafe24c7adbb087d1f03549")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{!"clang version 18.0.0 (ssh://git.vip.facebook.com/data/gitrepos/osmeta/external/llvm-project 680deb27e25976d9b74ad7b48ec5cb96be0e44b6)"}
!10 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 4, type: !11, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
!11 = !DISubroutineType(types: !12)
!12 = !{!13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{}
!15 = !DILocalVariable(name: "f", scope: !10, file: !1, line: 5, type: !16)
!16 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", file: !1, line: 1, size: 64, flags: DIFlagTypePassByValue, elements: !17, identifier: "_ZTS3Foo")
!17 = !{!18}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "c1", scope: !16, file: !1, line: 2, baseType: !19, size: 64)
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !20, size: 64)
!20 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!21 = !DILocation(line: 5, column: 6, scope: !10)
!22 = !DILocation(line: 6, column: 2, scope: !10)
