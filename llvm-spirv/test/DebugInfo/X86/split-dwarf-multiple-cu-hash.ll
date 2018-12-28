; RUN: rm -rf %t && mkdir -p %t
; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv -spirv-mem2reg=false
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o %t.ll

; RUN: llc -mtriple=%triple -split-dwarf-file=foo.dwo  %t.ll -filetype=obj -o %t/a.o
; RUN: llc -mtriple=%triple -split-dwarf-file=bar.dwo  %t.ll -filetype=obj -o %t/b.o
; RUN: llvm-dwarfdump -debug-info %t/a.o %t/b.o | FileCheck %s

; CHECK: .debug_info contents:
; CHECK: dwo_id {{.*}}([[HASH:.*]])
; CHECK-NOT: dwo_id {{.*}}([[HASH]])
; CHECK: .debug_info.dwo contents:

target triple = "spir64-unknown-unknown"

; Function Attrs: noinline nounwind uwtable
define void @_Z1av() #0 !dbg !9 {
entry:
  ret void, !dbg !12
}

; Function Attrs: noinline nounwind uwtable
define void @_Z1bv() #0 !dbg !13 {
entry:
  ret void, !dbg !14
}

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0, !3}
!llvm.ident = !{!5, !5}
!llvm.module.flags = !{!6, !7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 5.0.0 (trunk 304107) (llvm/trunk 304109)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "a.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch")
!2 = !{}
!3 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "clang version 5.0.0 (trunk 304107) (llvm/trunk 304109)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!4 = !DIFile(filename: "b.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch")
!5 = !{!"clang version 5.0.0 (trunk 304107) (llvm/trunk 304109)"}
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = distinct !DISubprogram(name: "a", linkageName: "_Z1av", scope: !1, file: !1, line: 1, type: !10, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = !DILocation(line: 2, column: 1, scope: !9)
!13 = distinct !DISubprogram(name: "b", linkageName: "_Z1bv", scope: !4, file: !4, line: 1, type: !10, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !3, retainedNodes: !2)
!14 = !DILocation(line: 2, column: 1, scope: !13)
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
