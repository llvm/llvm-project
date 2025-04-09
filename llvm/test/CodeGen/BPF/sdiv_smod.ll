; RUN: llc -mtriple=bpfel -mcpu=v4 -verify-machineinstrs -show-mc-encoding < %s | FileCheck %s
; Source:
;  int foo(int a, int b, int c) {
;    return a/b + a%c;
;  }
;  long bar(long a, long b, long c) {
;   return a/b + a%c;
; }
; Compilation flags:
;   clang -target bpf -O2 -S -emit-llvm t.c

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define dso_local i32 @foo(i32 noundef %a, i32 noundef %b, i32 noundef %c) local_unnamed_addr #0 {
entry:
  %div = sdiv i32 %a, %b
  %rem = srem i32 %a, %c
  %add = add nsw i32 %rem, %div
  ret i32 %add
}

; CHECK:       w0 = w1
; CHECK-NEXT:  w1 s/= w2                               # encoding: [0x3c,0x21,0x01,0x00,0x00,0x00,0x00,0x00]
; CHECK-NEXT:  w0 s%= w3                               # encoding: [0x9c,0x30,0x01,0x00,0x00,0x00,0x00,0x00]

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define dso_local i64 @bar(i64 noundef %a, i64 noundef %b, i64 noundef %c) local_unnamed_addr #0 {
entry:
  %div = sdiv i64 %a, %b
  %rem = srem i64 %a, %c
  %add = add nsw i64 %rem, %div
  ret i64 %add
}
; CHECK:       r0 = r1
; CHECK-NEXT:  r1 s/= r2                               # encoding: [0x3f,0x21,0x01,0x00,0x00,0x00,0x00,0x00]
; CHECK-NEXT:  r0 s%= r3                               # encoding: [0x9f,0x30,0x01,0x00,0x00,0x00,0x00,0x00]

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 17.0.0 (https://github.com/llvm/llvm-project.git c102025a4299e74767cdb4dfba8abbf6cbad820b)"}
