; RUN: llc -march=bpfel -mcpu=v4 -verify-machineinstrs -show-mc-encoding < %s | FileCheck %s
; Source:
;   short f1(int a) {
;     return (char)a;
;   }
;   int f2(int a) {
;     return (short)a;
;   }
;   long f3(int a) {
;     return (char)a;
;   }
;   long f4(int a) {
;     return (short)a;
;   }
;   long f5(int a) {
;     return a;
;   }
;   long f6(long a) {
;     return (int)a;
;   }
; Compilation flags:
;   clang -target bpf -O2 -mcpu=v4 -S -emit-llvm t.c

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define dso_local i16 @f1(i32 noundef %a) local_unnamed_addr #0 {
entry:
  %conv = trunc i32 %a to i8
  %conv1 = sext i8 %conv to i16
  ret i16 %conv1
}
; CHECK:  w0 = (s8)w1                             # encoding: [0xbc,0x10,0x08,0x00,0x00,0x00,0x00,0x00]

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define dso_local i32 @f2(i32 noundef %a) local_unnamed_addr #0 {
entry:
  %sext = shl i32 %a, 16
  %conv1 = ashr exact i32 %sext, 16
  ret i32 %conv1
}
; CHECK:  w0 = (s16)w1                            # encoding: [0xbc,0x10,0x10,0x00,0x00,0x00,0x00,0x00]

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define dso_local i64 @f3(i32 noundef %a) local_unnamed_addr #0 {
entry:
  %conv = zext i32 %a to i64
  %sext = shl i64 %conv, 56
  %conv1 = ashr exact i64 %sext, 56
  ret i64 %conv1
}
; CHECK:  r0 = (s8)r1                             # encoding: [0xbf,0x10,0x08,0x00,0x00,0x00,0x00,0x00]

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define dso_local i64 @f4(i32 noundef %a) local_unnamed_addr #0 {
entry:
  %conv = zext i32 %a to i64
  %sext = shl i64 %conv, 48
  %conv1 = ashr exact i64 %sext, 48
  ret i64 %conv1
}
; CHECK:  r0 = (s16)r1                            # encoding: [0xbf,0x10,0x10,0x00,0x00,0x00,0x00,0x00]

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define dso_local i64 @f5(i32 noundef %a) local_unnamed_addr #0 {
entry:
  %conv = sext i32 %a to i64
  ret i64 %conv
}
; CHECK:  r0 = (s32)r1                            # encoding: [0xbf,0x10,0x20,0x00,0x00,0x00,0x00,0x00]

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define dso_local i64 @f6(i64 noundef %a) local_unnamed_addr #0 {
entry:
  %sext = shl i64 %a, 32
  %conv1 = ashr exact i64 %sext, 32
  ret i64 %conv1
}
; CHECK:  r0 = (s32)r1                            # encoding: [0xbf,0x10,0x20,0x00,0x00,0x00,0x00,0x00]

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="v4" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 17.0.0 (https://github.com/llvm/llvm-project.git 4e1ca21db4162f0c3cde98f730b08ed538fff2a4)"}
