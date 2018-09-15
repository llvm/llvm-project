; Test to ensure that the LTO pipelines add pass to build the TargetLibraryInfo
; using the specified target triple.

; Check with regular LTO
; RUN: llvm-as < %s >%t1
; RUN: llvm-lto -exported-symbol=main -o %t2 %t1
; RUN: llvm-nm %t2 | FileCheck %s
; Check with ThinLTO. Use llvm-lto2 since this adds earlier passes requiring
; the TargetLibraryInfo with ThinLTO (WholeProgramDevirt).
; RUN: opt -module-summary -o %t1 %s
; RUN: llvm-lto2 run -r %t1,main,plx -o %t2 %t1
; RUN: llvm-nm %t2.1 | FileCheck %s

; We check that LTO will be aware of target triple and prevent exp2 to ldexpf
; transformation on Windows.
; CHECK: U exp2f

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.11.0"

define dso_local i32 @main(i32 %argc, i8** nocapture readnone %argv) local_unnamed_addr {
entry:
  %conv = sitofp i32 %argc to float
  %exp2 = tail call float @llvm.exp2.f32(float %conv)
  %conv1 = fptosi float %exp2 to i32
  ret i32 %conv1
}

; Function Attrs: nounwind readnone speculatable
declare float @llvm.exp2.f32(float)

