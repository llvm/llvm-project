; rdar://7860110
; RUN: llc -asm-verbose=false < %s | FileCheck %s -check-prefix=X64
; RUN: llc -march=x86 -asm-verbose=false < %s | FileCheck %s -check-prefix=X32
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.2"

define void @test1(i32* nocapture %a0, i8 zeroext %a1) nounwind ssp {
entry:
  %A = load i32, i32* %a0, align 4
  %B = and i32 %A, -256     ; 0xFFFFFF00
  %C = zext i8 %a1 to i32
  %D = or i32 %C, %B
  store i32 %D, i32* %a0, align 4
  ret void

; X64-LABEL: test1:
; X64: movb	%sil, (%rdi)

; X32-LABEL: test1:
; X32: movb	8(%esp), %al
; X32: movb	%al, (%{{.*}})
}

define void @test2(i32* nocapture %a0, i8 zeroext %a1) nounwind ssp {
entry:
  %A = load i32, i32* %a0, align 4
  %B = and i32 %A, -65281    ; 0xFFFF00FF
  %C = zext i8 %a1 to i32
  %CS = shl i32 %C, 8
  %D = or i32 %B, %CS
  store i32 %D, i32* %a0, align 4
  ret void
; X64-LABEL: test2:
; X64: movb	%sil, 1(%rdi)

; X32-LABEL: test2:
; X32: movb	8(%esp), %[[REG:[abcd]]]l
; X32: movb	%[[REG]]l, 1(%{{.*}})
}

define void @test3(i32* nocapture %a0, i16 zeroext %a1) nounwind ssp {
entry:
  %A = load i32, i32* %a0, align 4
  %B = and i32 %A, -65536    ; 0xFFFF0000
  %C = zext i16 %a1 to i32
  %D = or i32 %B, %C
  store i32 %D, i32* %a0, align 4
  ret void
; X64-LABEL: test3:
; X64: movw	%si, (%rdi)

; X32-LABEL: test3:
; X32: movw	8(%esp), %ax
; X32: movw	%ax, (%{{.*}})
}

define void @test4(i32* nocapture %a0, i16 zeroext %a1) nounwind ssp {
entry:
  %A = load i32, i32* %a0, align 4
  %B = and i32 %A, 65535    ; 0x0000FFFF
  %C = zext i16 %a1 to i32
  %CS = shl i32 %C, 16
  %D = or i32 %B, %CS
  store i32 %D, i32* %a0, align 4
  ret void
; X64-LABEL: test4:
; X64: movw	%si, 2(%rdi)

; X32-LABEL: test4:
; X32: movw	8(%esp), %[[REG:[abcd]]]x
; X32: movw	%[[REG]]x, 2(%{{.*}})
}

define void @test5(i64* nocapture %a0, i16 zeroext %a1) nounwind ssp {
entry:
  %A = load i64, i64* %a0, align 4
  %B = and i64 %A, -4294901761    ; 0xFFFFFFFF0000FFFF
  %C = zext i16 %a1 to i64
  %CS = shl i64 %C, 16
  %D = or i64 %B, %CS
  store i64 %D, i64* %a0, align 4
  ret void
; X64-LABEL: test5:
; X64: movw	%si, 2(%rdi)

; X32-LABEL: test5:
; X32: movw	8(%esp), %[[REG:[abcd]]]x
; X32: movw	%[[REG]]x, 2(%{{.*}})
}

define void @test6(i64* nocapture %a0, i8 zeroext %a1) nounwind ssp {
entry:
  %A = load i64, i64* %a0, align 4
  %B = and i64 %A, -280375465082881    ; 0xFFFF00FFFFFFFFFF
  %C = zext i8 %a1 to i64
  %CS = shl i64 %C, 40
  %D = or i64 %B, %CS
  store i64 %D, i64* %a0, align 4
  ret void
; X64-LABEL: test6:
; X64: movb	%sil, 5(%rdi)


; X32-LABEL: test6:
; X32: movb	8(%esp), %[[REG:[abcd]l]]
; X32: movb	%[[REG]], 5(%{{.*}})
}

define i32 @test7(i64* nocapture %a0, i8 zeroext %a1, i32* %P2) nounwind {
entry:
  %OtherLoad = load i32 , i32 *%P2
  %A = load i64, i64* %a0, align 4
  %B = and i64 %A, -280375465082881    ; 0xFFFF00FFFFFFFFFF
  %C = zext i8 %a1 to i64
  %CS = shl i64 %C, 40
  %D = or i64 %B, %CS
  store i64 %D, i64* %a0, align 4
  ret i32 %OtherLoad
; X64-LABEL: test7:
; X64: movb	%sil, 5(%rdi)


; X32-LABEL: test7:
; X32: movb	8(%esp), %[[REG:[abcd]l]]
; X32: movb	%[[REG]], 5(%{{.*}})
}

; PR7833

@g_16 = internal global i32 -1

; X64-LABEL: test8:
; X64-NEXT: movl _g_16(%rip), %eax
; X64-NEXT: movl $0, _g_16(%rip)
; X64-NEXT: orl  $1, %eax
; X64-NEXT: movl %eax, _g_16(%rip)
; X64-NEXT: ret
define void @test8() nounwind {
  %tmp = load i32, i32* @g_16
  store i32 0, i32* @g_16
  %or = or i32 %tmp, 1
  store i32 %or, i32* @g_16
  ret void
}

; X64-LABEL: test9:
; X64-NEXT: orb $1, _g_16(%rip)
; X64-NEXT: ret
define void @test9() nounwind {
  %tmp = load i32, i32* @g_16
  %or = or i32 %tmp, 1
  store i32 %or, i32* @g_16
  ret void
}

; rdar://8494845 + PR8244
; X64-LABEL: test10:
; X64-NEXT: movsbl	(%rdi), %eax
; X64-NEXT: shrl	$8, %eax
; X64-NEXT: ret
define i8 @test10(i8* %P) nounwind ssp {
entry:
  %tmp = load i8, i8* %P, align 1
  %conv = sext i8 %tmp to i32
  %shr3 = lshr i32 %conv, 8
  %conv2 = trunc i32 %shr3 to i8
  ret i8 %conv2
}
