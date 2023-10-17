; REQUIRES: system-linux
; RUN: llc -O0 --function-sections --generate-arange-section -filetype=obj %s -o %tmain.obj
; RUN: llvm-dwarfdump --debug-info %tmain.obj > %tmain.txt
; RUN: llvm-dwarfdump --debug-aranges %tmain.obj >> %tmain.txt
; RUN: cat %tmain.txt | FileCheck --check-prefix=CHECK %s

; CHECK: DW_AT_ranges
; CHECK-NEXT: [0x[[#%.16x,ADDR:]], 0x[[#%.16x,ADDR1:]])
; CHECK-NEXT: [0x[[#%.16x,ADDR2:]], 0x[[#%.16x,ADDR3:]])
; CHECK-NEXT: [0x[[#%.16x,ADDR4:]], 0x[[#%.16x,ADDR5:]])

; CHECK: Address Range Header
; CHECK-NEXT: [0x[[#ADDR]], 0x[[#ADDR1]])
; CHECK-NEXT: [0x[[#ADDR2]], 0x[[#ADDR3]])
; CHECK-NEXT: [0x[[#ADDR4]], 0x[[#ADDR5]])

; ModuleID = 'example.cpp'
source_filename = "example.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S = type { ptr }

@GlobalConst = dso_local global i32 42, align 4
@Global = dso_local global i32 0, align 4
@s = dso_local global %struct.S zeroinitializer, align 8

; Function Attrs: alwaysinline mustprogress nounwind uwtable
define dso_local noundef i32 @_Z6squarei(i32 noundef %i) #0 !dbg !10 {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, ptr %i.addr, align 4
  %0 = load i32, ptr %i.addr, align 4, !dbg !13
  %1 = load i32, ptr %i.addr, align 4, !dbg !14
  %mul = mul nsw i32 %0, %1, !dbg !15
  ret i32 %mul, !dbg !16
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local noundef i32 @_Z4cubei(i32 noundef %i) #1 !dbg !17 {
entry:
  %i.addr.i = alloca i32, align 4
  %i.addr = alloca i32, align 4
  %squared = alloca i32, align 4
  store i32 %i, ptr %i.addr, align 4
  %0 = load i32, ptr %i.addr, align 4, !dbg !18
  store i32 %0, ptr %i.addr.i, align 4
  %1 = load i32, ptr %i.addr.i, align 4, !dbg !19
  %2 = load i32, ptr %i.addr.i, align 4, !dbg !21
  %mul.i = mul nsw i32 %1, %2, !dbg !22
  store i32 %mul.i, ptr %squared, align 4, !dbg !23
  %3 = load i32, ptr %squared, align 4, !dbg !24
  %4 = load i32, ptr %i.addr, align 4, !dbg !25
  %mul = mul nsw i32 %3, %4, !dbg !26
  ret i32 %mul, !dbg !27
}

; Function Attrs: mustprogress noinline norecurse nounwind optnone uwtable
define dso_local noundef i32 @main() #2 !dbg !28 {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  ret i32 0, !dbg !29
}

attributes #0 = { alwaysinline mustprogress nounwind uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { mustprogress noinline norecurse nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 16.0.0 (ssh://git.vip.facebook.com/data/gitrepos/osmeta/external/llvm-project a312c5cce360203b4b24757e1e69738f8b3f2ec1)", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "example.cpp", directory: "/examples")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{!"clang version 16.0.0"}
!10 = distinct !DISubprogram(name: "square", scope: !1, file: !1, line: 11, type: !11, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !12)
!11 = !DISubroutineType(types: !12)
!12 = !{}
!13 = !DILocation(line: 11, column: 59, scope: !10)
!14 = !DILocation(line: 11, column: 63, scope: !10)
!15 = !DILocation(line: 11, column: 61, scope: !10)
!16 = !DILocation(line: 11, column: 52, scope: !10)
!17 = distinct !DISubprogram(name: "cube", scope: !1, file: !1, line: 12, type: !11, scopeLine: 12, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !12)
!18 = !DILocation(line: 13, column: 24, scope: !17)
!19 = !DILocation(line: 11, column: 59, scope: !10, inlinedAt: !20)
!20 = distinct !DILocation(line: 13, column: 17, scope: !17)
!21 = !DILocation(line: 11, column: 63, scope: !10, inlinedAt: !20)
!22 = !DILocation(line: 11, column: 61, scope: !10, inlinedAt: !20)
!23 = !DILocation(line: 13, column: 7, scope: !17)
!24 = !DILocation(line: 14, column: 10, scope: !17)
!25 = !DILocation(line: 14, column: 18, scope: !17)
!26 = !DILocation(line: 14, column: 17, scope: !17)
!27 = !DILocation(line: 14, column: 3, scope: !17)
!28 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 17, type: !11, scopeLine: 17, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !12)
!29 = !DILocation(line: 18, column: 3, scope: !28)
