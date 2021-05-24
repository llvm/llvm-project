; When using Itanium ABI, do not emit .debug_frame.
; RUNT: llc -mtriple=i386--linux -o - < %s | FileCheck %s -check-prefix=WITHOUT
; RUNT: llc -mtriple=armv7-netbsd-eabi -o - < %s | FileCheck %s -check-prefix=WITHOUT

; When using EHABI, do emit .debug_frame.
; RUN: llc -mtriple=arm-linux -mcpu=cortex-a7 -mattr=v7 -o - < %s | FileCheck %s -check-prefix=WITH

; REQUIRES: x86-registered-target
; REQUIRES: arm-registered-target

; WITH:        .cfi_sections .debug_frame
; WITHOUT-NOT: .cfi_sections

define i32 @foo() #0 !dbg !7 {
  %1 = call i32 @bar()
  %2 = call i32 @bar()
  %3 = add nsw i32 %1, %2
  ret i32 %3
}

declare i32 @bar() #1

attributes #0 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="arm7tdmi" "target-features"="+soft-float,+strict-align,-crypto,-neon" "unsafe-fp-math"="false" "use-soft-float"="true" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="arm7tdmi" "target-features"="+soft-float,+strict-align,-crypto,-neon" "unsafe-fp-math"="false" "use-soft-float"="true" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "cfi-sections.cc", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 1, !"min_enum_size", i32 4}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !8, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
