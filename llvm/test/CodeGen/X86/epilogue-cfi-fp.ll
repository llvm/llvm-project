; RUN: llc -O0 %s -o - | FileCheck %s

; ModuleID = 'epilogue-cfi-fp.c'
source_filename = "epilogue-cfi-fp.c"
target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i686-pc-linux"

; Function Attrs: noinline nounwind
define i32 @foo(i32 %i, i32 %j, i32 %k, i32 %l, i32 %m) #0 {

; CHECK-LABEL:   foo:
; CHECK:         popl %ebp
; CHECK-NEXT:    .cfi_def_cfa %esp, 4
; CHECK-NEXT:    retl

entry:
  %i.addr = alloca i32, align 4
  %j.addr = alloca i32, align 4
  %k.addr = alloca i32, align 4
  %l.addr = alloca i32, align 4
  %m.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  store i32 %j, i32* %j.addr, align 4
  store i32 %k, i32* %k.addr, align 4
  store i32 %l, i32* %l.addr, align 4
  store i32 %m, i32* %m.addr, align 4
  ret i32 0
}

attributes #0 = { "no-frame-pointer-elim"="true" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6, !7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 5.0.0 (http://llvm.org/git/clang.git 3f8116e6a2815b1d5f3491493938d0c63c9f42c9) (http://llvm.org/git/llvm.git 4fde77f8f1a8e4482e69b6a7484bc7d1b99b3c0a)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "epilogue-cfi-fp.c", directory: "epilogue-dwarf/test")
!2 = !{}
!3 = !{i32 1, !"NumRegisterParameters", i32 0}
!4 = !{i32 2, !"Dwarf Version", i32 4}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 1, !"wchar_size", i32 4}
!7 = !{i32 7, !"PIC Level", i32 2}

