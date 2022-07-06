; RUN: llc -mtriple=powerpc64le-unknown-unknown -mcpu=pwr8 -O0 < %s | FileCheck %s

; Just a test case for a crash reported in
; https://bugs.llvm.org/show_bug.cgi?id=33636

define void @main(i1 %arg) {
  %constexpr3 = zext i1 %arg to i32
  %constexpr4 = sdiv i32 1, %constexpr3
  %constexpr5 = trunc i32 %constexpr4 to i8
  %constexpr6 = icmp ne i8 %constexpr5, 0
  %constexpr8 = zext i1 %constexpr6 to i16
  store i16 %constexpr8, i16* null, align 2
  ret void
}

; CHECK: blr
