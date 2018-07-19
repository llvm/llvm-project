; RUN: llc -march=bpfel -filetype=obj -o - %s | llvm-objdump -d - | FileCheck --check-prefix=CHECK-EL %s
; RUN: llc -march=bpfeb -filetype=obj -o - %s | llvm-objdump -d - | FileCheck --check-prefix=CHECK-EB %s

; Function Attrs: nounwind uwtable
define i32 @ld_b(i64 %foo, i64* nocapture %bar, i8* %ctx, i8* %ctx2) #0 {
  %1 = tail call i64 @llvm.bpf.load.byte(i8* %ctx, i64 123) #2
  %2 = add i64 %1, %foo
  %3 = load volatile i64, i64* %bar, align 8
  %4 = add i64 %2, %3
  %5 = tail call i64 @llvm.bpf.load.byte(i8* %ctx2, i64 %foo) #2
  %6 = add i64 %4, %5
  %7 = load volatile i64, i64* %bar, align 8
  %8 = add i64 %6, %7
  %9 = trunc i64 %8 to i32
  ret i32 %9
; CHECK-LABEL: ld_b:
; CHECK-EL: r0 = *(u8 *)skb[123]
; CHECK-EL: r0 = *(u8 *)skb[r
; CHECK-EB: r0 = *(u8 *)skb[123]
; CHECK-EB: r0 = *(u8 *)skb[r
}

declare i64 @llvm.bpf.load.byte(i8*, i64) #1

; Function Attrs: nounwind uwtable
define i32 @ld_h(i8* %ctx, i8* %ctx2, i32 %foo) #0 {
  %1 = tail call i64 @llvm.bpf.load.half(i8* %ctx, i64 123) #2
  %2 = sext i32 %foo to i64
  %3 = tail call i64 @llvm.bpf.load.half(i8* %ctx2, i64 %2) #2
  %4 = add i64 %3, %1
  %5 = trunc i64 %4 to i32
  ret i32 %5
; CHECK-LABEL: ld_h:
; CHECK-EL: r0 = *(u16 *)skb[r
; CHECK-EL: r0 = *(u16 *)skb[123]
; CHECK-EB: r0 = *(u16 *)skb[r
; CHECK-EB: r0 = *(u16 *)skb[123]
}

declare i64 @llvm.bpf.load.half(i8*, i64) #1

; Function Attrs: nounwind uwtable
define i32 @ld_w(i8* %ctx, i8* %ctx2, i32 %foo) #0 {
  %1 = tail call i64 @llvm.bpf.load.word(i8* %ctx, i64 123) #2
  %2 = sext i32 %foo to i64
  %3 = tail call i64 @llvm.bpf.load.word(i8* %ctx2, i64 %2) #2
  %4 = add i64 %3, %1
  %5 = trunc i64 %4 to i32
  ret i32 %5
; CHECK-LABEL: ld_w:
; CHECK-EL: r0 = *(u32 *)skb[r
; CHECK-EL: r0 = *(u32 *)skb[123]
; CHECK-EB: r0 = *(u32 *)skb[r
; CHECK-EB: r0 = *(u32 *)skb[123]
}

declare i64 @llvm.bpf.load.word(i8*, i64) #1

define i32 @ld_pseudo() #0 {
entry:
  %call = tail call i64 @llvm.bpf.pseudo(i64 2, i64 3)
  tail call void inttoptr (i64 4 to void (i64, i32)*)(i64 %call, i32 4) #2
  ret i32 0
; CHECK-LABEL: ld_pseudo:
; CHECK-EL: ld_pseudo r1, 2, 3
; CHECK-EB: ld_pseudo r1, 2, 3
}

declare i64 @llvm.bpf.pseudo(i64, i64) #2

define i32 @bswap(i64 %a, i64 %b, i64 %c) #0 {
entry:
  %0 = tail call i64 @llvm.bswap.i64(i64 %a)
  %conv = trunc i64 %b to i32
  %1 = tail call i32 @llvm.bswap.i32(i32 %conv)
  %conv1 = zext i32 %1 to i64
  %add = add i64 %conv1, %0
  %conv2 = trunc i64 %c to i16
  %2 = tail call i16 @llvm.bswap.i16(i16 %conv2)
  %conv3 = zext i16 %2 to i64
  %add4 = add i64 %add, %conv3
  %conv5 = trunc i64 %add4 to i32
  ret i32 %conv5
; CHECK-LABEL: bswap:
; CHECK-EL: r1 = be64 r1
; CHECK-EL: r2 = be32 r2
; CHECK-EL: r2 += r1
; CHECK-EL: r3 = be16 r3
; CHECK-EL: r2 += r3
; CHECK-EB: r1 = le64 r1
; CHECK-EB: r2 = le32 r2
; CHECK-EB: r2 += r1
; CHECK-EB: r3 = le16 r3
; CHECK-EB: r2 += r3
}

declare i64 @llvm.bswap.i64(i64) #1
declare i32 @llvm.bswap.i32(i32) #1
declare i16 @llvm.bswap.i16(i16) #1
