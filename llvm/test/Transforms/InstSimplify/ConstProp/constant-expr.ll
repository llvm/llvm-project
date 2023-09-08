; RUN: llvm-as < %s | llvm-dis | FileCheck %s

@X = external global i8
@Y = external global i8
@Z = external global i8

@A = global i1 add (i1 icmp ult (ptr @X, ptr @Y), i1 icmp ult (ptr @X, ptr @Z))
; CHECK: @A = global i1 xor (i1 icmp ult (ptr @X, ptr @Y), i1 icmp ult (ptr @X, ptr @Z))
@B = global i1 sub (i1 icmp ult (ptr @X, ptr @Y), i1 icmp ult (ptr @X, ptr @Z)), align 2
; CHECK: @B = global i1 xor (i1 icmp ult (ptr @X, ptr @Y), i1 icmp ult (ptr @X, ptr @Z))
@C = global i1 mul (i1 icmp ult (ptr @X, ptr @Y), i1 icmp ult (ptr @X, ptr @Z))
; CHECK: @C = global i1 mul (i1 icmp ult (ptr @X, ptr @Y), i1 icmp ult (ptr @X, ptr @Z))

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
