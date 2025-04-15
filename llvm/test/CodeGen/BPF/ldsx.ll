; RUN: llc -mtriple=bpfel -mcpu=v4 -verify-machineinstrs -show-mc-encoding < %s | FileCheck %s
; Source:
;  int f1(char *p) {
;    return *p;
;  }
;  int f2(short *p) {
;    return *p;
;  }
;  int f3(int *p) {
;    return *p;
;  }
;  long f4(char *p) {
;    return *p;
;  }
;  long f5(short *p) {
;    return *p;
;  }
;  long f6(int *p) {
;    return *p;
;  }
;  long f7(long *p) {
;    return *p;
;  }
; Compilation flags:
;   clang -target bpf -O2 -S -emit-llvm t.c

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read)
define dso_local i32 @f1(ptr nocapture noundef readonly %p) local_unnamed_addr #0 {
entry:
  %0 = load i8, ptr %p, align 1, !tbaa !3
  %conv = sext i8 %0 to i32
  ret i32 %conv
}
; CHECK:  r0 = *(s8 *)(r1 + 0)                    # encoding: [0x91,0x10,0x00,0x00,0x00,0x00,0x00,0x00]

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read)
define dso_local i32 @f2(ptr nocapture noundef readonly %p) local_unnamed_addr #0 {
entry:
  %0 = load i16, ptr %p, align 2, !tbaa !6
  %conv = sext i16 %0 to i32
  ret i32 %conv
}
; CHECK:  r0 = *(s16 *)(r1 + 0)                   # encoding: [0x89,0x10,0x00,0x00,0x00,0x00,0x00,0x00]

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read)
define dso_local i32 @f3(ptr nocapture noundef readonly %p) local_unnamed_addr #0 {
entry:
  %0 = load i32, ptr %p, align 4, !tbaa !8
  ret i32 %0
}
; CHECK:  w0 = *(u32 *)(r1 + 0)                   # encoding: [0x61,0x10,0x00,0x00,0x00,0x00,0x00,0x00]

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read)
define dso_local i64 @f4(ptr nocapture noundef readonly %p) local_unnamed_addr #0 {
entry:
  %0 = load i8, ptr %p, align 1, !tbaa !3
  %conv = sext i8 %0 to i64
  ret i64 %conv
}
; CHECK:  r0 = *(s8 *)(r1 + 0)                    # encoding: [0x91,0x10,0x00,0x00,0x00,0x00,0x00,0x00]

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read)
define dso_local i64 @f5(ptr nocapture noundef readonly %p) local_unnamed_addr #0 {
entry:
  %0 = load i16, ptr %p, align 2, !tbaa !6
  %conv = sext i16 %0 to i64
  ret i64 %conv
}
; CHECK:  r0 = *(s16 *)(r1 + 0)                   # encoding: [0x89,0x10,0x00,0x00,0x00,0x00,0x00,0x00]

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read)
define dso_local i64 @f6(ptr nocapture noundef readonly %p) local_unnamed_addr #0 {
entry:
  %0 = load i32, ptr %p, align 4, !tbaa !8
  %conv = sext i32 %0 to i64
  ret i64 %conv
}
; CHECK:  r0 = *(s32 *)(r1 + 0)                   # encoding: [0x81,0x10,0x00,0x00,0x00,0x00,0x00,0x00]

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read)
define dso_local i64 @f7(ptr nocapture noundef readonly %p) local_unnamed_addr #0 {
entry:
  %0 = load i64, ptr %p, align 8, !tbaa !10
  ret i64 %0
}
; CHECK:  r0 = *(u64 *)(r1 + 0)                   # encoding: [0x79,0x10,0x00,0x00,0x00,0x00,0x00,0x00]

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 17.0.0 (https://github.com/llvm/llvm-project.git 1bf3221bf1e35d953a0b6783bc6e694cb9b0ceae)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"short", !4, i64 0}
!8 = !{!9, !9, i64 0}
!9 = !{!"int", !4, i64 0}
!10 = !{!11, !11, i64 0}
!11 = !{!"long", !4, i64 0}
