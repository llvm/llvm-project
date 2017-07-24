; RUN: llc -mtriple=x86_64-mingw32 < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-linux < %s | FileCheck %s
; CHECK-NOT: -{{[1-9][0-9]*}}(%rsp)

define win64cc x86_fp80 @a(i64 %x) nounwind readnone {
entry:
        %conv = sitofp i64 %x to x86_fp80               ; <x86_fp80> [#uses=1]
        ret x86_fp80 %conv
}
