; RUN: llvm-as < %s | llvm-dis | FileCheck %s

@X = external global i8
@Y = external global i8
@Z = external global i8

@A = global i1 add (i1 icmp ult (ptr @X, ptr @Y), i1 icmp ult (ptr @X, ptr @Z))
; CHECK: @A = global i1 xor (i1 icmp ult (ptr @X, ptr @Y), i1 icmp ult (ptr @X, ptr @Z))
@B = global i1 sub (i1 icmp ult (ptr @X, ptr @Y), i1 icmp ult (ptr @X, ptr @Z)), align 2
; CHECK: @B = global i1 xor (i1 icmp ult (ptr @X, ptr @Y), i1 icmp ult (ptr @X, ptr @Z))
@C = global i1 mul (i1 icmp ult (ptr @X, ptr @Y), i1 icmp ult (ptr @X, ptr @Z))
; CHECK: @C = global i1 and (i1 icmp ult (ptr @X, ptr @Y), i1 icmp ult (ptr @X, ptr @Z))

@D = global i1 sdiv (i1 icmp ult (ptr @X, ptr @Y), i1 icmp ult (ptr @X, ptr @Z))
; CHECK: @D = global i1 icmp ult (ptr @X, ptr @Y)
@E = global i1 udiv (i1 icmp ult (ptr @X, ptr @Y), i1 icmp ult (ptr @X, ptr @Z))
; CHECK: @E = global i1 icmp ult (ptr @X, ptr @Y)
@F = global i1 srem (i1 icmp ult (ptr @X, ptr @Y), i1 icmp ult (ptr @X, ptr @Z))
; CHECK: @F = global i1 false 
@G = global i1 urem (i1 icmp ult (ptr @X, ptr @Y), i1 icmp ult (ptr @X, ptr @Z))
; CHECK: @G = global i1 false 

@H = global i1 icmp ule (ptr @X, ptr @Y)
; CHECK: @H = global i1 icmp ule (ptr @X, ptr @Y)

@I = global i1 xor (i1 icmp ult (ptr @X, ptr @Y), i1 false)
; CHECK: @I = global i1 icmp ult (ptr @X, ptr @Y)
@J = global i1 xor (i1 icmp ult (ptr @X, ptr @Y), i1 true)
; CHECK: @J = global i1 icmp uge (ptr @X, ptr @Y)

@K = global i1 icmp eq (i1 icmp ult (ptr @X, ptr @Y), i1 false)
; CHECK: @K = global i1 icmp uge (ptr @X, ptr @Y)
@L = global i1 icmp eq (i1 icmp ult (ptr @X, ptr @Y), i1 true)
; CHECK: @L = global i1 icmp ult (ptr @X, ptr @Y)
@M = global i1 icmp ne (i1 icmp ult (ptr @X, ptr @Y), i1 true)
; CHECK: @M = global i1 icmp uge (ptr @X, ptr @Y)
@N = global i1 icmp ne (i1 icmp ult (ptr @X, ptr @Y), i1 false)
; CHECK: @N = global i1 icmp ult (ptr @X, ptr @Y)

@O = global i1 icmp eq (i32 zext (i1 icmp ult (ptr @X, ptr @Y) to i32), i32 0)
; CHECK: @O = global i1 icmp uge (ptr @X, ptr @Y)



; PR5176

; CHECK: @T1 = global i1 true
@T1 = global i1 icmp eq (i64 and (i64 trunc (i256 lshr (i256 or (i256 and (i256 and (i256 shl (i256 zext (i64 ptrtoint (ptr @B to i64) to i256), i256 64), i256 -6277101735386680763495507056286727952638980837032266301441), i256 6277101735386680763835789423207666416102355444464034512895), i256 shl (i256 zext (i64 ptrtoint (ptr @A to i64) to i256), i256 192)), i256 64) to i64), i64 1), i64 0)

; CHECK: @T2 = global ptr @B
@T2 = global ptr inttoptr (i64 add (i64 trunc (i256 lshr (i256 or (i256 and (i256 and (i256 shl (i256 zext (i64 ptrtoint (ptr @A to i64) to i256), i256 64), i256 -6277101735386680763495507056286727952638980837032266301441), i256 6277101735386680763835789423207666416102355444464034512895), i256 shl (i256 zext (i64 ptrtoint (ptr @B to i64) to i256), i256 192)), i256 192) to i64), i64 trunc (i256 lshr (i256 or (i256 and (i256 and (i256 shl (i256 zext (i64 ptrtoint (ptr @A to i64) to i256), i256 64), i256 -6277101735386680763495507056286727952638980837032266301441), i256 6277101735386680763835789423207666416102355444464034512895), i256 shl (i256 zext (i64 ptrtoint (ptr @B to i64) to i256), i256 192)), i256 128) to i64)) to ptr)

; CHECK: @T3 = global i64 add (i64 ptrtoint (ptr @B to i64), i64 -1)
@T3 = global i64 add (i64 trunc (i256 lshr (i256 or (i256 and (i256 and (i256 shl (i256 zext (i64 ptrtoint (ptr @B to i64) to i256), i256 64), i256 -6277101735386680763495507056286727952638980837032266301441), i256 6277101735386680763835789423207666416102355444464034512895), i256 shl (i256 zext (i64 ptrtoint (ptr @A to i64) to i256), i256 192)), i256 64) to i64), i64 -1)

; CHECK: @T4 = global ptr @B
@T4 = global ptr inttoptr (i64 trunc (i256 lshr (i256 or (i256 and (i256 and (i256 shl (i256 zext (i64 ptrtoint (ptr @B to i64) to i256), i256 64), i256 -6277101735386680763495507056286727952638980837032266301441), i256 6277101735386680763835789423207666416102355444464034512895), i256 shl (i256 zext (i64 ptrtoint (ptr @A to i64) to i256), i256 192)), i256 64) to i64) to ptr)

; CHECK: @T5 = global ptr @A
@T5 = global ptr inttoptr (i64 add (i64 trunc (i256 lshr (i256 or (i256 and (i256 and (i256 shl (i256 zext (i64 ptrtoint (ptr @B to i64) to i256), i256 64), i256 -6277101735386680763495507056286727952638980837032266301441), i256 6277101735386680763835789423207666416102355444464034512895), i256 shl (i256 zext (i64 ptrtoint (ptr @A to i64) to i256), i256 192)), i256 192) to i64), i64 trunc (i256 lshr (i256 or (i256 and (i256 and (i256 shl (i256 zext (i64 ptrtoint (ptr @B to i64) to i256), i256 64), i256 -6277101735386680763495507056286727952638980837032266301441), i256 6277101735386680763835789423207666416102355444464034512895), i256 shl (i256 zext (i64 ptrtoint (ptr @A to i64) to i256), i256 192)), i256 128) to i64)) to ptr)



; PR6096

; No check line. This used to crash llvm-as.
@T6 = global <2 x i1> fcmp ole (<2 x float> fdiv (<2 x float> undef, <2 x float> <float 1.000000e+00, float 1.000000e+00>), <2 x float> zeroinitializer)


; PR9011

@pr9011_1 = constant <4 x i32> zext (<4 x i8> zeroinitializer to <4 x i32>)
; CHECK: pr9011_1 = constant <4 x i32> zeroinitializer
@pr9011_2 = constant <4 x i32> sext (<4 x i8> zeroinitializer to <4 x i32>)
; CHECK: pr9011_2 = constant <4 x i32> zeroinitializer
@pr9011_3 = constant <4 x i32> bitcast (<16 x i8> zeroinitializer to <4 x i32>)
; CHECK: pr9011_3 = constant <4 x i32> zeroinitializer
@pr9011_4 = constant <4 x float> uitofp (<4 x i8> zeroinitializer to <4 x float>)
; CHECK: pr9011_4 = constant <4 x float> zeroinitializer
@pr9011_5 = constant <4 x float> sitofp (<4 x i8> zeroinitializer to <4 x float>)
; CHECK: pr9011_5 = constant <4 x float> zeroinitializer
@pr9011_6 = constant <4 x i32> fptosi (<4 x float> zeroinitializer to <4 x i32>)
; CHECK: pr9011_6 = constant <4 x i32> zeroinitializer
@pr9011_7 = constant <4 x i32> fptoui (<4 x float> zeroinitializer to <4 x i32>)
; CHECK: pr9011_7 = constant <4 x i32> zeroinitializer
@pr9011_8 = constant <4 x float> fptrunc (<4 x double> zeroinitializer to <4 x float>)
; CHECK: pr9011_8 = constant <4 x float> zeroinitializer
@pr9011_9 = constant <4 x double> fpext (<4 x float> zeroinitializer to <4 x double>)
; CHECK: pr9011_9 = constant <4 x double> zeroinitializer

@pr9011_10 = constant <4 x double> bitcast (i256 0 to <4 x double>)
; CHECK: pr9011_10 = constant <4 x double> zeroinitializer
@pr9011_11 = constant <4 x float> bitcast (i128 0 to <4 x float>)
; CHECK: pr9011_11 = constant <4 x float> zeroinitializer
@pr9011_12 = constant <4 x i32> bitcast (i128 0 to <4 x i32>)
; CHECK: pr9011_12 = constant <4 x i32> zeroinitializer
@pr9011_13 = constant i256 bitcast (<4 x double> zeroinitializer to i256)
; CHECK: pr9011_13 = constant i256 0
@pr9011_14 = constant i128 bitcast (<4 x float> zeroinitializer to i128)
; CHECK: pr9011_14 = constant i128 0
@pr9011_15 = constant i128 bitcast (<4 x i32> zeroinitializer to i128)
; CHECK: pr9011_15 = constant i128 0

@select = internal constant
          i32 select (i1 icmp ult (i32 ptrtoint (ptr @X to i32),
                                   i32 ptrtoint (ptr @Y to i32)),
            i32 select (i1 icmp ult (i32 ptrtoint (ptr @X to i32),
                                     i32 ptrtoint (ptr @Y to i32)),
               i32 10, i32 20),
            i32 30)
; CHECK: select = internal constant i32 select {{.*}} i32 10, i32 30
