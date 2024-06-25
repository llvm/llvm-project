; REQUIRES: aarch64-registered-target
; This test makes sure that the debug_frame section does not contain any named symbols.
; The ll file was compiled by using the following command.
; > clang -arch arm64 -ffreestanding -g -c -x c -o /tmp/test.o /tmp/test.c
;
; > cat /tmp/test.c
; void foo(void) {}


; RUN: llc %s -O0 -filetype=obj -o - | llvm-nm -s __DWARF __debug_frame - | FileCheck %s --allow-empty

; CHECK-NOT: {{.*}}ltmp{{[0-9]+}}

; ModuleID = '/tmp/test.c'
source_filename = "/tmp/test.c"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx14.0.0"

; Function Attrs: noinline nounwind optnone ssp
define void @foo() #0 !dbg !8 {
entry:
  ret void, !dbg !12
}

attributes #0 = { noinline nounwind optnone ssp "frame-pointer"="non-leaf" "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a,+zcm,+zcz" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 17.0.0 (git@github.com:llvm/llvm-project.git e734a12b608f8c4a2b03fb2f3194de1cc3b43344)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
!1 = !DIFile(filename: "/tmp/test.c", directory: "/tmp")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"frame-pointer", i32 1}
!7 = !{!"clang version 17.0.0 (git@github.com:llvm/llvm-project.git e734a12b608f8c4a2b03fb2f3194de1cc3b43344)"}
!8 = distinct !DISubprogram(name: "foo", scope: !9, file: !9, line: 1, type: !10, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DIFile(filename: "/tmp/test.c", directory: "")
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = !DILocation(line: 1, column: 17, scope: !8)
