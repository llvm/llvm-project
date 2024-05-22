; RUN: llc -march=bpfel -mcpu=v4 -verify-machineinstrs -show-mc-encoding < %s | FileCheck %s
; Source:
;  long foo(int a, int b, long c) {
;    a = __builtin_bswap16(a);
;    b = __builtin_bswap32(b);
;    c = __builtin_bswap64(c);
;    return a + b + c;
;  }
; Compilation flags:
;   clang -target bpf -O2 -S -emit-llvm t.c

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
define dso_local i64 @foo(i32 noundef %a, i32 noundef %b, i64 noundef %c) local_unnamed_addr #0 {
entry:
  %conv = trunc i32 %a to i16
  %0 = tail call i16 @llvm.bswap.i16(i16 %conv)
  %conv1 = zext i16 %0 to i32
  %1 = tail call i32 @llvm.bswap.i32(i32 %b)
  %2 = tail call i64 @llvm.bswap.i64(i64 %c)
  %add = add nsw i32 %1, %conv1
  %conv2 = sext i32 %add to i64
  %add3 = add nsw i64 %2, %conv2
  ret i64 %add3
}

; CHECK: r1 = bswap16 r1                         # encoding: [0xd7,0x01,0x00,0x00,0x10,0x00,0x00,0x00]
; CHECK: r2 = bswap32 r2                         # encoding: [0xd7,0x02,0x00,0x00,0x20,0x00,0x00,0x00]
; CHECK: r0 = bswap64 r0                         # encoding: [0xd7,0x00,0x00,0x00,0x40,0x00,0x00,0x00]

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i16 @llvm.bswap.i16(i16) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.bswap.i32(i32) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.bswap.i64(i64) #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(none) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 17.0.0 (https://github.com/llvm/llvm-project.git a2913a8a2bfe572d2f1bfea950ab9b0848373648)"}
