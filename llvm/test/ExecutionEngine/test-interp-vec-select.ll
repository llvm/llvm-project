; RUN: %lli -force-interpreter=true %s > /dev/null

define i32 @main() {

  ; Vector values
  %a2_i8 = add <2 x i8> zeroinitializer, <i8 0, i8 1>
  %a3_i8 = add <3 x i8> zeroinitializer, <i8 0, i8 1, i8 2>
  %a4_i8 = add <4 x i8> zeroinitializer, <i8 0, i8 1, i8 2, i8 3>
  %a8_i8 = add <8 x i8> zeroinitializer, <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7>
  %a16_i8 = add <16 x i8> zeroinitializer, <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>

  %a2_i16 = add <2 x i16> zeroinitializer, <i16 0, i16 1>
  %a3_i16 = add <3 x i16> zeroinitializer, <i16 0, i16 1, i16 2>
  %a4_i16 = add <4 x i16> zeroinitializer, <i16 0, i16 1, i16 2, i16 3>
  %a8_i16 = add <8 x i16> zeroinitializer, <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>
  %a16_i16 = add <16 x i16> zeroinitializer, <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7, i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>

  %a2_i32 = add <2 x i32> zeroinitializer, <i32 0, i32 1>
  %a3_i32 = add <3 x i32> zeroinitializer, <i32 0, i32 1, i32 2>
  %a4_i32 = add <4 x i32> zeroinitializer, <i32 0, i32 1, i32 2, i32 3>
  %a8_i32 = add <8 x i32> zeroinitializer, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %a16_i32 = add <16 x i32> zeroinitializer, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

  %a2_i64 = add <2 x i64> zeroinitializer, <i64 0, i64 1>
  %a3_i64 = add <3 x i64> zeroinitializer, <i64 0, i64 1, i64 2>
  %a4_i64 = add <4 x i64> zeroinitializer, <i64 0, i64 1, i64 2, i64 3>
  %a8_i64 = add <8 x i64> zeroinitializer, <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7>
  %a16_i64 = add <16 x i64> zeroinitializer, <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9, i64 10, i64 11, i64 12, i64 13, i64 14, i64 15>

  %a2_float = fadd <2 x float> zeroinitializer, <float 0.0, float 1.0>
  %a3_float = fadd <3 x float> zeroinitializer, <float 0.0, float 1.0, float 2.0>
  %a4_float = fadd <4 x float> zeroinitializer, <float 0.0, float 1.0, float 2.0, float 3.0>
  %a8_float = fadd <8 x float> zeroinitializer, <float 0.0, float 1.0, float 2.0, float 3.0, float 4.0, float 5.0, float 6.0, float 7.0>
  %a16_float = fadd <16 x float> zeroinitializer, <float 0.0, float 1.0, float 2.0, float 3.0, float 4.0, float 5.0, float 6.0, float 7.0, float 8.0, float 9.0, float 10.0, float 11.0, float 12.0, float 13.0, float 14.0, float 15.0>

  %a2_double = fadd <2 x double> zeroinitializer, <double 0.0, double 1.0>
  %a3_double = fadd <3 x double> zeroinitializer, <double 0.0, double 1.0, double 2.0>
  %a4_double = fadd <4 x double> zeroinitializer, <double 0.0, double 1.0, double 2.0, double 3.0>
  %a8_double = fadd <8 x double> zeroinitializer, <double 0.0, double 1.0, double 2.0, double 3.0, double 4.0, double 5.0, double 6.0, double 7.0>
  %a16_double = fadd <16 x double> zeroinitializer, <double 0.0, double 1.0, double 2.0, double 3.0, double 4.0, double 5.0, double 6.0, double 7.0, double 8.0, double 9.0, double 10.0, double 11.0, double 12.0, double 13.0, double 14.0, double 15.0>

  %b2_i8  = sub <2 x i8> zeroinitializer, %a2_i8
  %b3_i8  = sub <3 x i8> zeroinitializer, %a3_i8
  %b4_i8  = sub <4 x i8> zeroinitializer, %a4_i8
  %b8_i8  = sub <8 x i8> zeroinitializer, %a8_i8
  %b16_i8 = sub <16 x i8> zeroinitializer, %a16_i8

  %b2_i16  = sub <2 x i16> zeroinitializer, %a2_i16
  %b3_i16  = sub <3 x i16> zeroinitializer, %a3_i16
  %b4_i16  = sub <4 x i16> zeroinitializer, %a4_i16
  %b8_i16  = sub <8 x i16> zeroinitializer, %a8_i16
  %b16_i16 = sub <16 x i16> zeroinitializer, %a16_i16

  %b2_i32  = sub <2 x i32> zeroinitializer, %a2_i32
  %b3_i32  = sub <3 x i32> zeroinitializer, %a3_i32
  %b4_i32  = sub <4 x i32> zeroinitializer, %a4_i32
  %b8_i32  = sub <8 x i32> zeroinitializer, %a8_i32
  %b16_i32 = sub <16 x i32> zeroinitializer, %a16_i32

  %b2_i64  = sub <2 x i64> zeroinitializer, %a2_i64
  %b3_i64  = sub <3 x i64> zeroinitializer, %a3_i64
  %b4_i64  = sub <4 x i64> zeroinitializer, %a4_i64
  %b8_i64  = sub <8 x i64> zeroinitializer, %a8_i64
  %b16_i64 = sub <16 x i64> zeroinitializer, %a16_i64

  %b2_float  = fsub <2 x float> zeroinitializer, %a2_float
  %b3_float  = fsub <3 x float> zeroinitializer, %a3_float
  %b4_float  = fsub <4 x float> zeroinitializer, %a4_float
  %b8_float  = fsub <8 x float> zeroinitializer, %a8_float
  %b16_float = fsub <16 x float> zeroinitializer, %a16_float

  %b2_double  = fsub <2 x double> zeroinitializer, %a2_double
  %b3_double  = fsub <3 x double> zeroinitializer, %a3_double
  %b4_double  = fsub <4 x double> zeroinitializer, %a4_double
  %b8_double  = fsub <8 x double> zeroinitializer, %a8_double
  %b16_double = fsub <16 x double> zeroinitializer, %a16_double



  %v0 = select <2 x i1> <i1 true, i1 false>, <2 x i8> %a2_i8, <2 x i8> %b2_i8
  %v1 = select <3 x i1> <i1 true, i1 false, i1 true>, <3 x i8> %a3_i8, <3 x i8> %b3_i8
  %v2 = select <4 x i1> <i1 true, i1 false, i1 true, i1 false>, <4 x i8> %a4_i8, <4 x i8> %b4_i8
  %v3 = select <8 x i1> <i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false>, <8 x i8> %a8_i8, <8 x i8> %b8_i8
  %v4 = select <16 x i1> <i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false>, <16 x i8> %a16_i8, <16 x i8> %b16_i8

  %v5 = select <2 x i1> <i1 true, i1 false>, <2 x i16> %a2_i16, <2 x i16> %b2_i16
  %v6 = select <3 x i1> <i1 true, i1 false, i1 true>, <3 x i16> %a3_i16, <3 x i16> %b3_i16
  %v7 = select <4 x i1> <i1 true, i1 false, i1 true, i1 false>, <4 x i16> %a4_i16, <4 x i16> %b4_i16
  %v8 = select <8 x i1> <i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false>, <8 x i16> %a8_i16, <8 x i16> %b8_i16
  %v9 = select <16 x i1> <i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false>, <16 x i16> %a16_i16, <16 x i16> %b16_i16

  %v10 = select <2 x i1> <i1 true, i1 false>, <2 x i32> %a2_i32, <2 x i32> %b2_i32
  %v11 = select <3 x i1> <i1 true, i1 false, i1 true>, <3 x i32> %a3_i32, <3 x i32> %b3_i32
  %v12 = select <4 x i1> <i1 true, i1 false, i1 true, i1 false>, <4 x i32> %a4_i32, <4 x i32> %b4_i32
  %v13 = select <8 x i1> <i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false>, <8 x i32> %a8_i32, <8 x i32> %b8_i32
  %v14 = select <16 x i1> <i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false>, <16 x i32> %a16_i32, <16 x i32> %b16_i32

  %v15 = select <2 x i1> <i1 true, i1 false>, <2 x i64> %a2_i64, <2 x i64> %b2_i64
  %v16 = select <3 x i1> <i1 true, i1 false, i1 true>, <3 x i64> %a3_i64, <3 x i64> %b3_i64
  %v17 = select <4 x i1> <i1 true, i1 false, i1 true, i1 false>, <4 x i64> %a4_i64, <4 x i64> %b4_i64
  %v18 = select <8 x i1> <i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false>, <8 x i64> %a8_i64, <8 x i64> %b8_i64
  %v19 = select <16 x i1> <i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false>, <16 x i64> %a16_i64, <16 x i64> %b16_i64

  %v20 = select <2 x i1> <i1 true, i1 false>, <2 x float> %a2_float, <2 x float> %b2_float
  %v21 = select <3 x i1> <i1 true, i1 false, i1 true>, <3 x float> %a3_float, <3 x float> %b3_float
  %v22 = select <4 x i1> <i1 true, i1 false, i1 true, i1 false>, <4 x float> %a4_float, <4 x float> %b4_float
  %v23 = select <8 x i1> <i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false>, <8 x float> %a8_float, <8 x float> %b8_float
  %v24 = select <16 x i1> <i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false>, <16 x float> %a16_float, <16 x float> %b16_float

  %v25 = select <2 x i1> <i1 true, i1 false>, <2 x double> %a2_double, <2 x double> %b2_double
  %v26 = select <3 x i1> <i1 true, i1 false, i1 true>, <3 x double> %a3_double, <3 x double> %b3_double
  %v27 = select <4 x i1> <i1 true, i1 false, i1 true, i1 false>, <4 x double> %a4_double, <4 x double> %b4_double
  %v28 = select <8 x i1> <i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false>, <8 x double> %a8_double, <8 x double> %b8_double
  %v29 = select <16 x i1> <i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false>, <16 x double> %a16_double, <16 x double> %b16_double


  ret i32 0
}
