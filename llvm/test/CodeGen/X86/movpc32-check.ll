; RUN: llc < %s -mtriple=i686-pc-linux -relocation-model=pic | FileCheck %s

target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i686-pc-linux"

; Function Attrs: nounwind
define void @test() #0 !dbg !4 {
entry:
  call void bitcast (void (...)* @bar to void ()*)(), !dbg !11
  ret void, !dbg !12
}

declare void @bar(...) #1

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="i686" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="i686" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.0 (http://llvm.org/git/clang.git 3490ab8630d5643f71f1f04e46984f05b27b8d67) (http://llvm.org/git/llvm.git d2643e2ff955ed234944fe3c6b4ffc1250085843)", isOptimized: false, runtimeVersion: 0, emissionKind: 1, enums: !2, subprograms: !3)
!1 = !DIFile(filename: "test.c", directory: "movpc-test")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 2, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, isOptimized: false, variables: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"PIC Level", i32 2}
!10 = !{!"clang version 3.8.0 (http://llvm.org/git/clang.git 3490ab8630d5643f71f1f04e46984f05b27b8d67) (http://llvm.org/git/llvm.git d2643e2ff955ed234944fe3c6b4ffc1250085843)"}
!11 = !DILocation(line: 4, column: 3, scope: !4)
!12 = !DILocation(line: 5, column: 1, scope: !4)

; CHECK: calll .L0$pb
; CHECK-NEXT: .Ltmp3:
; CHECK-NEXT: .cfi_adjust_cfa_offset 4
; CHECK-NEXT: .L0$pb:
; CHECK-NEXT: popl
; CHECK-NEXT: .Ltmp4:
; CHECK-NEXT: .cfi_adjust_cfa_offset -4
