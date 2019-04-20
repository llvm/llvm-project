; RUN: llc -O0 < %s | FileCheck %s
; FIXME: Add test for llc with optimizations once it is implemented.

; Source to regenerate:
; $ clang --target=x86_64-windows-msvc -S heapallocsite.c  -g -gcodeview -o t.ll \
;      -emit-llvm -O0 -Xclang -disable-llvm-passes -fms-extensions
; __declspec(allocator) char *myalloc();
; void g();
; void foo() {
;   g();
;   myalloc()
;   g();
; }

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-msvc"

; Function Attrs: noinline nounwind optnone
define dso_local void @f() #0 !dbg !7 {
entry:
  call void @g(), !dbg !11
  %call = call i8* @myalloc(), !dbg !12, !heapallocsite !13
  call void @g(), !dbg !14
  ret void, !dbg !15
}

; CHECK-LABEL: f: # @f
; CHECK: callq g
; CHECK: .Lheapallocsite0:
; CHECK: callq myalloc
; CHECK: .Lheapallocsite1:
; CHECK: retq

; CHECK-LABEL: .short  4423                    # Record kind: S_GPROC32_ID
; CHECK:       .short  4446                    # Record kind: S_HEAPALLOCSITE
; CHECK-NEXT:  .secrel32       .Lheapallocsite0
; CHECK-NEXT:  .secidx .Lheapallocsite0
; CHECK-NEXT:  .short  .Lheapallocsite1-.Lheapallocsite0
; CHECK-NEXT:  .long 112
; CHECK-NEXT:  .p2align 2

; CHECK-LABEL: .short  4431                    # Record kind: S_PROC_ID_END

declare dso_local void @g() #1

declare dso_local i8* @myalloc() #1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0 (https://github.com/llvm/llvm-project.git 4eff3de99423a62fd6e833e29c71c1e62ba6140b)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "C:\5Csrc\5Ctest", checksumkind: CSK_MD5, checksum: "6d758cfa3834154a04ce8a55102772a9")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{!"clang version 9.0.0 (https://github.com/llvm/llvm-project.git 4eff3de99423a62fd6e833e29c71c1e62ba6140b)"}
!7 = distinct !DISubprogram(name: "f", scope: !8, file: !8, line: 4, type: !9, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DIFile(filename: "heapallocsite.c", directory: "C:\5Csrc\5Ctest", checksumkind: CSK_MD5, checksum: "6d758cfa3834154a04ce8a55102772a9")
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DILocation(line: 5, scope: !7)
!12 = !DILocation(line: 6, scope: !7)
!13 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!14 = !DILocation(line: 7, scope: !7)
!15 = !DILocation(line: 8, scope: !7)

