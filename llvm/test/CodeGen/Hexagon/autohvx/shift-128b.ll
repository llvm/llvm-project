; RUN: llc -march=hexagon < %s | FileCheck %s

; CHECK-LABEL: test0000:
; CHECK: v0.h = vasl(v0.h,r0)
define <64 x i16> @test0000(<64 x i16> %a0, i16 %a1) #0 {
  %b0 = insertelement <64 x i16> zeroinitializer, i16 %a1, i32 0
  %b1 = insertelement <64 x i16> %b0, i16 %a1, i32 1
  %b2 = insertelement <64 x i16> %b1, i16 %a1, i32 2
  %b3 = insertelement <64 x i16> %b2, i16 %a1, i32 3
  %b4 = insertelement <64 x i16> %b3, i16 %a1, i32 4
  %b5 = insertelement <64 x i16> %b4, i16 %a1, i32 5
  %b6 = insertelement <64 x i16> %b5, i16 %a1, i32 6
  %b7 = insertelement <64 x i16> %b6, i16 %a1, i32 7
  %b8 = insertelement <64 x i16> %b7, i16 %a1, i32 8
  %b9 = insertelement <64 x i16> %b8, i16 %a1, i32 9
  %b10 = insertelement <64 x i16> %b9, i16 %a1, i32 10
  %b11 = insertelement <64 x i16> %b10, i16 %a1, i32 11
  %b12 = insertelement <64 x i16> %b11, i16 %a1, i32 12
  %b13 = insertelement <64 x i16> %b12, i16 %a1, i32 13
  %b14 = insertelement <64 x i16> %b13, i16 %a1, i32 14
  %b15 = insertelement <64 x i16> %b14, i16 %a1, i32 15
  %b16 = insertelement <64 x i16> %b15, i16 %a1, i32 16
  %b17 = insertelement <64 x i16> %b16, i16 %a1, i32 17
  %b18 = insertelement <64 x i16> %b17, i16 %a1, i32 18
  %b19 = insertelement <64 x i16> %b18, i16 %a1, i32 19
  %b20 = insertelement <64 x i16> %b19, i16 %a1, i32 20
  %b21 = insertelement <64 x i16> %b20, i16 %a1, i32 21
  %b22 = insertelement <64 x i16> %b21, i16 %a1, i32 22
  %b23 = insertelement <64 x i16> %b22, i16 %a1, i32 23
  %b24 = insertelement <64 x i16> %b23, i16 %a1, i32 24
  %b25 = insertelement <64 x i16> %b24, i16 %a1, i32 25
  %b26 = insertelement <64 x i16> %b25, i16 %a1, i32 26
  %b27 = insertelement <64 x i16> %b26, i16 %a1, i32 27
  %b28 = insertelement <64 x i16> %b27, i16 %a1, i32 28
  %b29 = insertelement <64 x i16> %b28, i16 %a1, i32 29
  %b30 = insertelement <64 x i16> %b29, i16 %a1, i32 30
  %b31 = insertelement <64 x i16> %b30, i16 %a1, i32 31
  %b32 = insertelement <64 x i16> %b31, i16 %a1, i32 32
  %b33 = insertelement <64 x i16> %b32, i16 %a1, i32 33
  %b34 = insertelement <64 x i16> %b33, i16 %a1, i32 34
  %b35 = insertelement <64 x i16> %b34, i16 %a1, i32 35
  %b36 = insertelement <64 x i16> %b35, i16 %a1, i32 36
  %b37 = insertelement <64 x i16> %b36, i16 %a1, i32 37
  %b38 = insertelement <64 x i16> %b37, i16 %a1, i32 38
  %b39 = insertelement <64 x i16> %b38, i16 %a1, i32 39
  %b40 = insertelement <64 x i16> %b39, i16 %a1, i32 40
  %b41 = insertelement <64 x i16> %b40, i16 %a1, i32 41
  %b42 = insertelement <64 x i16> %b41, i16 %a1, i32 42
  %b43 = insertelement <64 x i16> %b42, i16 %a1, i32 43
  %b44 = insertelement <64 x i16> %b43, i16 %a1, i32 44
  %b45 = insertelement <64 x i16> %b44, i16 %a1, i32 45
  %b46 = insertelement <64 x i16> %b45, i16 %a1, i32 46
  %b47 = insertelement <64 x i16> %b46, i16 %a1, i32 47
  %b48 = insertelement <64 x i16> %b47, i16 %a1, i32 48
  %b49 = insertelement <64 x i16> %b48, i16 %a1, i32 49
  %b50 = insertelement <64 x i16> %b49, i16 %a1, i32 50
  %b51 = insertelement <64 x i16> %b50, i16 %a1, i32 51
  %b52 = insertelement <64 x i16> %b51, i16 %a1, i32 52
  %b53 = insertelement <64 x i16> %b52, i16 %a1, i32 53
  %b54 = insertelement <64 x i16> %b53, i16 %a1, i32 54
  %b55 = insertelement <64 x i16> %b54, i16 %a1, i32 55
  %b56 = insertelement <64 x i16> %b55, i16 %a1, i32 56
  %b57 = insertelement <64 x i16> %b56, i16 %a1, i32 57
  %b58 = insertelement <64 x i16> %b57, i16 %a1, i32 58
  %b59 = insertelement <64 x i16> %b58, i16 %a1, i32 59
  %b60 = insertelement <64 x i16> %b59, i16 %a1, i32 60
  %b61 = insertelement <64 x i16> %b60, i16 %a1, i32 61
  %b62 = insertelement <64 x i16> %b61, i16 %a1, i32 62
  %b63 = insertelement <64 x i16> %b62, i16 %a1, i32 63
  %v0 = shl <64 x i16> %a0, %b63
  ret <64 x i16> %v0
}

; CHECK-LABEL: test0001:
; CHECK: v0.h = vasr(v0.h,r0)
define <64 x i16> @test0001(<64 x i16> %a0, i16 %a1) #0 {
  %b0 = insertelement <64 x i16> zeroinitializer, i16 %a1, i32 0
  %b1 = insertelement <64 x i16> %b0, i16 %a1, i32 1
  %b2 = insertelement <64 x i16> %b1, i16 %a1, i32 2
  %b3 = insertelement <64 x i16> %b2, i16 %a1, i32 3
  %b4 = insertelement <64 x i16> %b3, i16 %a1, i32 4
  %b5 = insertelement <64 x i16> %b4, i16 %a1, i32 5
  %b6 = insertelement <64 x i16> %b5, i16 %a1, i32 6
  %b7 = insertelement <64 x i16> %b6, i16 %a1, i32 7
  %b8 = insertelement <64 x i16> %b7, i16 %a1, i32 8
  %b9 = insertelement <64 x i16> %b8, i16 %a1, i32 9
  %b10 = insertelement <64 x i16> %b9, i16 %a1, i32 10
  %b11 = insertelement <64 x i16> %b10, i16 %a1, i32 11
  %b12 = insertelement <64 x i16> %b11, i16 %a1, i32 12
  %b13 = insertelement <64 x i16> %b12, i16 %a1, i32 13
  %b14 = insertelement <64 x i16> %b13, i16 %a1, i32 14
  %b15 = insertelement <64 x i16> %b14, i16 %a1, i32 15
  %b16 = insertelement <64 x i16> %b15, i16 %a1, i32 16
  %b17 = insertelement <64 x i16> %b16, i16 %a1, i32 17
  %b18 = insertelement <64 x i16> %b17, i16 %a1, i32 18
  %b19 = insertelement <64 x i16> %b18, i16 %a1, i32 19
  %b20 = insertelement <64 x i16> %b19, i16 %a1, i32 20
  %b21 = insertelement <64 x i16> %b20, i16 %a1, i32 21
  %b22 = insertelement <64 x i16> %b21, i16 %a1, i32 22
  %b23 = insertelement <64 x i16> %b22, i16 %a1, i32 23
  %b24 = insertelement <64 x i16> %b23, i16 %a1, i32 24
  %b25 = insertelement <64 x i16> %b24, i16 %a1, i32 25
  %b26 = insertelement <64 x i16> %b25, i16 %a1, i32 26
  %b27 = insertelement <64 x i16> %b26, i16 %a1, i32 27
  %b28 = insertelement <64 x i16> %b27, i16 %a1, i32 28
  %b29 = insertelement <64 x i16> %b28, i16 %a1, i32 29
  %b30 = insertelement <64 x i16> %b29, i16 %a1, i32 30
  %b31 = insertelement <64 x i16> %b30, i16 %a1, i32 31
  %b32 = insertelement <64 x i16> %b31, i16 %a1, i32 32
  %b33 = insertelement <64 x i16> %b32, i16 %a1, i32 33
  %b34 = insertelement <64 x i16> %b33, i16 %a1, i32 34
  %b35 = insertelement <64 x i16> %b34, i16 %a1, i32 35
  %b36 = insertelement <64 x i16> %b35, i16 %a1, i32 36
  %b37 = insertelement <64 x i16> %b36, i16 %a1, i32 37
  %b38 = insertelement <64 x i16> %b37, i16 %a1, i32 38
  %b39 = insertelement <64 x i16> %b38, i16 %a1, i32 39
  %b40 = insertelement <64 x i16> %b39, i16 %a1, i32 40
  %b41 = insertelement <64 x i16> %b40, i16 %a1, i32 41
  %b42 = insertelement <64 x i16> %b41, i16 %a1, i32 42
  %b43 = insertelement <64 x i16> %b42, i16 %a1, i32 43
  %b44 = insertelement <64 x i16> %b43, i16 %a1, i32 44
  %b45 = insertelement <64 x i16> %b44, i16 %a1, i32 45
  %b46 = insertelement <64 x i16> %b45, i16 %a1, i32 46
  %b47 = insertelement <64 x i16> %b46, i16 %a1, i32 47
  %b48 = insertelement <64 x i16> %b47, i16 %a1, i32 48
  %b49 = insertelement <64 x i16> %b48, i16 %a1, i32 49
  %b50 = insertelement <64 x i16> %b49, i16 %a1, i32 50
  %b51 = insertelement <64 x i16> %b50, i16 %a1, i32 51
  %b52 = insertelement <64 x i16> %b51, i16 %a1, i32 52
  %b53 = insertelement <64 x i16> %b52, i16 %a1, i32 53
  %b54 = insertelement <64 x i16> %b53, i16 %a1, i32 54
  %b55 = insertelement <64 x i16> %b54, i16 %a1, i32 55
  %b56 = insertelement <64 x i16> %b55, i16 %a1, i32 56
  %b57 = insertelement <64 x i16> %b56, i16 %a1, i32 57
  %b58 = insertelement <64 x i16> %b57, i16 %a1, i32 58
  %b59 = insertelement <64 x i16> %b58, i16 %a1, i32 59
  %b60 = insertelement <64 x i16> %b59, i16 %a1, i32 60
  %b61 = insertelement <64 x i16> %b60, i16 %a1, i32 61
  %b62 = insertelement <64 x i16> %b61, i16 %a1, i32 62
  %b63 = insertelement <64 x i16> %b62, i16 %a1, i32 63
  %v0 = ashr <64 x i16> %a0, %b63
  ret <64 x i16> %v0
}

; CHECK-LABEL: test0002:
; CHECK: v0.uh = vlsr(v0.uh,r0)
define <64 x i16> @test0002(<64 x i16> %a0, i16 %a1) #0 {
  %b0 = insertelement <64 x i16> zeroinitializer, i16 %a1, i32 0
  %b1 = insertelement <64 x i16> %b0, i16 %a1, i32 1
  %b2 = insertelement <64 x i16> %b1, i16 %a1, i32 2
  %b3 = insertelement <64 x i16> %b2, i16 %a1, i32 3
  %b4 = insertelement <64 x i16> %b3, i16 %a1, i32 4
  %b5 = insertelement <64 x i16> %b4, i16 %a1, i32 5
  %b6 = insertelement <64 x i16> %b5, i16 %a1, i32 6
  %b7 = insertelement <64 x i16> %b6, i16 %a1, i32 7
  %b8 = insertelement <64 x i16> %b7, i16 %a1, i32 8
  %b9 = insertelement <64 x i16> %b8, i16 %a1, i32 9
  %b10 = insertelement <64 x i16> %b9, i16 %a1, i32 10
  %b11 = insertelement <64 x i16> %b10, i16 %a1, i32 11
  %b12 = insertelement <64 x i16> %b11, i16 %a1, i32 12
  %b13 = insertelement <64 x i16> %b12, i16 %a1, i32 13
  %b14 = insertelement <64 x i16> %b13, i16 %a1, i32 14
  %b15 = insertelement <64 x i16> %b14, i16 %a1, i32 15
  %b16 = insertelement <64 x i16> %b15, i16 %a1, i32 16
  %b17 = insertelement <64 x i16> %b16, i16 %a1, i32 17
  %b18 = insertelement <64 x i16> %b17, i16 %a1, i32 18
  %b19 = insertelement <64 x i16> %b18, i16 %a1, i32 19
  %b20 = insertelement <64 x i16> %b19, i16 %a1, i32 20
  %b21 = insertelement <64 x i16> %b20, i16 %a1, i32 21
  %b22 = insertelement <64 x i16> %b21, i16 %a1, i32 22
  %b23 = insertelement <64 x i16> %b22, i16 %a1, i32 23
  %b24 = insertelement <64 x i16> %b23, i16 %a1, i32 24
  %b25 = insertelement <64 x i16> %b24, i16 %a1, i32 25
  %b26 = insertelement <64 x i16> %b25, i16 %a1, i32 26
  %b27 = insertelement <64 x i16> %b26, i16 %a1, i32 27
  %b28 = insertelement <64 x i16> %b27, i16 %a1, i32 28
  %b29 = insertelement <64 x i16> %b28, i16 %a1, i32 29
  %b30 = insertelement <64 x i16> %b29, i16 %a1, i32 30
  %b31 = insertelement <64 x i16> %b30, i16 %a1, i32 31
  %b32 = insertelement <64 x i16> %b31, i16 %a1, i32 32
  %b33 = insertelement <64 x i16> %b32, i16 %a1, i32 33
  %b34 = insertelement <64 x i16> %b33, i16 %a1, i32 34
  %b35 = insertelement <64 x i16> %b34, i16 %a1, i32 35
  %b36 = insertelement <64 x i16> %b35, i16 %a1, i32 36
  %b37 = insertelement <64 x i16> %b36, i16 %a1, i32 37
  %b38 = insertelement <64 x i16> %b37, i16 %a1, i32 38
  %b39 = insertelement <64 x i16> %b38, i16 %a1, i32 39
  %b40 = insertelement <64 x i16> %b39, i16 %a1, i32 40
  %b41 = insertelement <64 x i16> %b40, i16 %a1, i32 41
  %b42 = insertelement <64 x i16> %b41, i16 %a1, i32 42
  %b43 = insertelement <64 x i16> %b42, i16 %a1, i32 43
  %b44 = insertelement <64 x i16> %b43, i16 %a1, i32 44
  %b45 = insertelement <64 x i16> %b44, i16 %a1, i32 45
  %b46 = insertelement <64 x i16> %b45, i16 %a1, i32 46
  %b47 = insertelement <64 x i16> %b46, i16 %a1, i32 47
  %b48 = insertelement <64 x i16> %b47, i16 %a1, i32 48
  %b49 = insertelement <64 x i16> %b48, i16 %a1, i32 49
  %b50 = insertelement <64 x i16> %b49, i16 %a1, i32 50
  %b51 = insertelement <64 x i16> %b50, i16 %a1, i32 51
  %b52 = insertelement <64 x i16> %b51, i16 %a1, i32 52
  %b53 = insertelement <64 x i16> %b52, i16 %a1, i32 53
  %b54 = insertelement <64 x i16> %b53, i16 %a1, i32 54
  %b55 = insertelement <64 x i16> %b54, i16 %a1, i32 55
  %b56 = insertelement <64 x i16> %b55, i16 %a1, i32 56
  %b57 = insertelement <64 x i16> %b56, i16 %a1, i32 57
  %b58 = insertelement <64 x i16> %b57, i16 %a1, i32 58
  %b59 = insertelement <64 x i16> %b58, i16 %a1, i32 59
  %b60 = insertelement <64 x i16> %b59, i16 %a1, i32 60
  %b61 = insertelement <64 x i16> %b60, i16 %a1, i32 61
  %b62 = insertelement <64 x i16> %b61, i16 %a1, i32 62
  %b63 = insertelement <64 x i16> %b62, i16 %a1, i32 63
  %v0 = lshr <64 x i16> %a0, %b63
  ret <64 x i16> %v0
}

; CHECK-LABEL: test0010:
; CHECK: v0.w = vasl(v0.w,r0)
define <32 x i32> @test0010(<32 x i32> %a0, i32 %a1) #0 {
  %b0 = insertelement <32 x i32> zeroinitializer, i32 %a1, i32 0
  %b1 = insertelement <32 x i32> %b0, i32 %a1, i32 1
  %b2 = insertelement <32 x i32> %b1, i32 %a1, i32 2
  %b3 = insertelement <32 x i32> %b2, i32 %a1, i32 3
  %b4 = insertelement <32 x i32> %b3, i32 %a1, i32 4
  %b5 = insertelement <32 x i32> %b4, i32 %a1, i32 5
  %b6 = insertelement <32 x i32> %b5, i32 %a1, i32 6
  %b7 = insertelement <32 x i32> %b6, i32 %a1, i32 7
  %b8 = insertelement <32 x i32> %b7, i32 %a1, i32 8
  %b9 = insertelement <32 x i32> %b8, i32 %a1, i32 9
  %b10 = insertelement <32 x i32> %b9, i32 %a1, i32 10
  %b11 = insertelement <32 x i32> %b10, i32 %a1, i32 11
  %b12 = insertelement <32 x i32> %b11, i32 %a1, i32 12
  %b13 = insertelement <32 x i32> %b12, i32 %a1, i32 13
  %b14 = insertelement <32 x i32> %b13, i32 %a1, i32 14
  %b15 = insertelement <32 x i32> %b14, i32 %a1, i32 15
  %b16 = insertelement <32 x i32> %b15, i32 %a1, i32 16
  %b17 = insertelement <32 x i32> %b16, i32 %a1, i32 17
  %b18 = insertelement <32 x i32> %b17, i32 %a1, i32 18
  %b19 = insertelement <32 x i32> %b18, i32 %a1, i32 19
  %b20 = insertelement <32 x i32> %b19, i32 %a1, i32 20
  %b21 = insertelement <32 x i32> %b20, i32 %a1, i32 21
  %b22 = insertelement <32 x i32> %b21, i32 %a1, i32 22
  %b23 = insertelement <32 x i32> %b22, i32 %a1, i32 23
  %b24 = insertelement <32 x i32> %b23, i32 %a1, i32 24
  %b25 = insertelement <32 x i32> %b24, i32 %a1, i32 25
  %b26 = insertelement <32 x i32> %b25, i32 %a1, i32 26
  %b27 = insertelement <32 x i32> %b26, i32 %a1, i32 27
  %b28 = insertelement <32 x i32> %b27, i32 %a1, i32 28
  %b29 = insertelement <32 x i32> %b28, i32 %a1, i32 29
  %b30 = insertelement <32 x i32> %b29, i32 %a1, i32 30
  %b31 = insertelement <32 x i32> %b30, i32 %a1, i32 31
  %v0 = shl <32 x i32> %a0, %b31
  ret <32 x i32> %v0
}

; CHECK-LABEL: test0011:
; CHECK: v0.w = vasr(v0.w,r0)
define <32 x i32> @test0011(<32 x i32> %a0, i32 %a1) #0 {
  %b0 = insertelement <32 x i32> zeroinitializer, i32 %a1, i32 0
  %b1 = insertelement <32 x i32> %b0, i32 %a1, i32 1
  %b2 = insertelement <32 x i32> %b1, i32 %a1, i32 2
  %b3 = insertelement <32 x i32> %b2, i32 %a1, i32 3
  %b4 = insertelement <32 x i32> %b3, i32 %a1, i32 4
  %b5 = insertelement <32 x i32> %b4, i32 %a1, i32 5
  %b6 = insertelement <32 x i32> %b5, i32 %a1, i32 6
  %b7 = insertelement <32 x i32> %b6, i32 %a1, i32 7
  %b8 = insertelement <32 x i32> %b7, i32 %a1, i32 8
  %b9 = insertelement <32 x i32> %b8, i32 %a1, i32 9
  %b10 = insertelement <32 x i32> %b9, i32 %a1, i32 10
  %b11 = insertelement <32 x i32> %b10, i32 %a1, i32 11
  %b12 = insertelement <32 x i32> %b11, i32 %a1, i32 12
  %b13 = insertelement <32 x i32> %b12, i32 %a1, i32 13
  %b14 = insertelement <32 x i32> %b13, i32 %a1, i32 14
  %b15 = insertelement <32 x i32> %b14, i32 %a1, i32 15
  %b16 = insertelement <32 x i32> %b15, i32 %a1, i32 16
  %b17 = insertelement <32 x i32> %b16, i32 %a1, i32 17
  %b18 = insertelement <32 x i32> %b17, i32 %a1, i32 18
  %b19 = insertelement <32 x i32> %b18, i32 %a1, i32 19
  %b20 = insertelement <32 x i32> %b19, i32 %a1, i32 20
  %b21 = insertelement <32 x i32> %b20, i32 %a1, i32 21
  %b22 = insertelement <32 x i32> %b21, i32 %a1, i32 22
  %b23 = insertelement <32 x i32> %b22, i32 %a1, i32 23
  %b24 = insertelement <32 x i32> %b23, i32 %a1, i32 24
  %b25 = insertelement <32 x i32> %b24, i32 %a1, i32 25
  %b26 = insertelement <32 x i32> %b25, i32 %a1, i32 26
  %b27 = insertelement <32 x i32> %b26, i32 %a1, i32 27
  %b28 = insertelement <32 x i32> %b27, i32 %a1, i32 28
  %b29 = insertelement <32 x i32> %b28, i32 %a1, i32 29
  %b30 = insertelement <32 x i32> %b29, i32 %a1, i32 30
  %b31 = insertelement <32 x i32> %b30, i32 %a1, i32 31
  %v0 = ashr <32 x i32> %a0, %b31
  ret <32 x i32> %v0
}

; CHECK-LABEL: test0012:
; CHECK: v0.uw = vlsr(v0.uw,r0)
define <32 x i32> @test0012(<32 x i32> %a0, i32 %a1) #0 {
  %b0 = insertelement <32 x i32> zeroinitializer, i32 %a1, i32 0
  %b1 = insertelement <32 x i32> %b0, i32 %a1, i32 1
  %b2 = insertelement <32 x i32> %b1, i32 %a1, i32 2
  %b3 = insertelement <32 x i32> %b2, i32 %a1, i32 3
  %b4 = insertelement <32 x i32> %b3, i32 %a1, i32 4
  %b5 = insertelement <32 x i32> %b4, i32 %a1, i32 5
  %b6 = insertelement <32 x i32> %b5, i32 %a1, i32 6
  %b7 = insertelement <32 x i32> %b6, i32 %a1, i32 7
  %b8 = insertelement <32 x i32> %b7, i32 %a1, i32 8
  %b9 = insertelement <32 x i32> %b8, i32 %a1, i32 9
  %b10 = insertelement <32 x i32> %b9, i32 %a1, i32 10
  %b11 = insertelement <32 x i32> %b10, i32 %a1, i32 11
  %b12 = insertelement <32 x i32> %b11, i32 %a1, i32 12
  %b13 = insertelement <32 x i32> %b12, i32 %a1, i32 13
  %b14 = insertelement <32 x i32> %b13, i32 %a1, i32 14
  %b15 = insertelement <32 x i32> %b14, i32 %a1, i32 15
  %b16 = insertelement <32 x i32> %b15, i32 %a1, i32 16
  %b17 = insertelement <32 x i32> %b16, i32 %a1, i32 17
  %b18 = insertelement <32 x i32> %b17, i32 %a1, i32 18
  %b19 = insertelement <32 x i32> %b18, i32 %a1, i32 19
  %b20 = insertelement <32 x i32> %b19, i32 %a1, i32 20
  %b21 = insertelement <32 x i32> %b20, i32 %a1, i32 21
  %b22 = insertelement <32 x i32> %b21, i32 %a1, i32 22
  %b23 = insertelement <32 x i32> %b22, i32 %a1, i32 23
  %b24 = insertelement <32 x i32> %b23, i32 %a1, i32 24
  %b25 = insertelement <32 x i32> %b24, i32 %a1, i32 25
  %b26 = insertelement <32 x i32> %b25, i32 %a1, i32 26
  %b27 = insertelement <32 x i32> %b26, i32 %a1, i32 27
  %b28 = insertelement <32 x i32> %b27, i32 %a1, i32 28
  %b29 = insertelement <32 x i32> %b28, i32 %a1, i32 29
  %b30 = insertelement <32 x i32> %b29, i32 %a1, i32 30
  %b31 = insertelement <32 x i32> %b30, i32 %a1, i32 31
  %v0 = lshr <32 x i32> %a0, %b31
  ret <32 x i32> %v0
}

; CHECK-LABEL: test0020:
; CHECK: v0.h = vasl(v0.h,v1.h)
define <64 x i16> @test0020(<64 x i16> %a0, <64 x i16> %a1) #0 {
  %v0 = shl <64 x i16> %a0, %a1
  ret <64 x i16> %v0
}

; CHECK-LABEL: test0021:
; CHECK: v0.h = vasr(v0.h,v1.h)
define <64 x i16> @test0021(<64 x i16> %a0, <64 x i16> %a1) #0 {
  %v0 = ashr <64 x i16> %a0, %a1
  ret <64 x i16> %v0
}

; CHECK-LABEL: test0022:
; CHECK: v0.h = vlsr(v0.h,v1.h)
define <64 x i16> @test0022(<64 x i16> %a0, <64 x i16> %a1) #0 {
  %v0 = lshr <64 x i16> %a0, %a1
  ret <64 x i16> %v0
}

; CHECK-LABEL: test0030:
; CHECK: v0.w = vasl(v0.w,v1.w)
define <32 x i32> @test0030(<32 x i32> %a0, <32 x i32> %a1) #0 {
  %v0 = shl <32 x i32> %a0, %a1
  ret <32 x i32> %v0
}

; CHECK-LABEL: test0031:
; CHECK: v0.w = vasr(v0.w,v1.w)
define <32 x i32> @test0031(<32 x i32> %a0, <32 x i32> %a1) #0 {
  %v0 = ashr <32 x i32> %a0, %a1
  ret <32 x i32> %v0
}

; CHECK-LABEL: test0032:
; CHECK: v0.w = vlsr(v0.w,v1.w)
define <32 x i32> @test0032(<32 x i32> %a0, <32 x i32> %a1) #0 {
  %v0 = lshr <32 x i32> %a0, %a1
  ret <32 x i32> %v0
}

attributes #0 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length128b" }

