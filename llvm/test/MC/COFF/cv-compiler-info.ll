; RUN: llc -mtriple i686-pc-windows-msvc < %s | FileCheck %s
; ModuleID = 'D:\src\scopes\foo.cpp'
source_filename = "D:\5Csrc\5Cscopes\5Cfoo.cpp"
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc19.0.23918"

; Function Attrs: nounwind sspstrong
define i32 @"\01?foo@@YAHXZ"() #0 !dbg !10 {
entry:
  ret i32 42, !dbg !14
}

attributes #0 = { nounwind sspstrong "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 4.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
; One .debug$S section should contain an S_COMPILE3 record that identifies the
; source language and the version of the compiler based on the DICompileUnit.
; CHECK: 	.section	.debug$S,"dr"
; CHECK: 		.short	4412                  # Record kind: S_COMPILE3
; CHECK: 		.long	1                       # Flags and language
; CHECK: 		.short	7                     # CPUType
; CHECK: 		.short	4                     # Frontend version
; CHECK: 		.short	0
; CHECK: 		.short	0
; CHECK: 		.short	0
; CHECK: 		.short	[[BACKEND_VERSION:[0-9]+]]  # Backend version
; CHECK: 		.short	0
; CHECK: 		.short	0
; CHECK: 		.short	0
; CHECK: 		.asciz	"clang version 4.0.0 "  # Null-terminated compiler version string
; CHECK-NOT: .short	4412                  # Record kind: S_COMPILE3
!1 = !DIFile(filename: "D:\5Csrc\5Cscopes\5Cfoo.cpp", directory: "D:\5Csrc\5Cscopes\5Cclang")
!2 = !{}
!7 = !{i32 2, !"CodeView", i32 1}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{!"clang version 4.0.0 "}
!10 = distinct !DISubprogram(name: "foo", linkageName: "\01?foo@@YAHXZ", scope: !1, file: !1, line: 1, type: !11, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!11 = !DISubroutineType(types: !12)
!12 = !{!13}
!13 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!14 = !DILocation(line: 2, scope: !10)
