 ; RUN: %lli -jit-kind=mcjit -force-interpreter=true %s > /dev/null

define i32 @main() {
 %v0 = insertelement <2 x i8> zeroinitializer, i8 1, i32 1
 %v1 = insertelement <3 x i8> zeroinitializer, i8 2, i32 2
 %v2 = insertelement <4 x i8> zeroinitializer, i8 3, i32 3
 %v3 = insertelement <8 x i8> zeroinitializer, i8 4, i32 4
 %v4 = insertelement <16 x i8> zeroinitializer, i8 5, i32 7

 %v5 = insertelement <2 x i16> zeroinitializer, i16 1, i32 1
 %v6 = insertelement <3 x i16> zeroinitializer, i16 2, i32 2
 %v7 = insertelement <4 x i16> zeroinitializer, i16 3, i32 3
 %v8 = insertelement <8 x i16> zeroinitializer, i16 4, i32 4
 %v9 = insertelement <16 x i16> zeroinitializer, i16 5, i32 7

 %v10 = insertelement <2 x i32> zeroinitializer, i32 1, i32 1
 %v11 = insertelement <3 x i32> zeroinitializer, i32 2, i32 2
 %v12 = insertelement <4 x i32> zeroinitializer, i32 3, i32 3
 %v13 = insertelement <8 x i32> zeroinitializer, i32 4, i32 4
 %v14 = insertelement <16 x i32> zeroinitializer, i32 5, i32 7

 %v15 = insertelement <2 x i64> zeroinitializer, i64 1, i32 1
 %v16 = insertelement <3 x i64> zeroinitializer, i64 2, i32 2
 %v17 = insertelement <4 x i64> zeroinitializer, i64 3, i32 3
 %v18 = insertelement <8 x i64> zeroinitializer, i64 4, i32 4
 %v19 = insertelement <16 x i64> zeroinitializer, i64 5, i32 7

 %v20 = insertelement <2 x float> zeroinitializer, float 1.0, i32 1
 %v21 = insertelement <3 x float> zeroinitializer, float 2.0, i32 2
 %v22 = insertelement <4 x float> zeroinitializer, float 3.0, i32 3
 %v23 = insertelement <8 x float> zeroinitializer, float 4.0, i32 4
 %v24 = insertelement <16 x float> zeroinitializer, float 5.0, i32 7

 %v25 = insertelement <2 x double> zeroinitializer, double 1.0, i32 1
 %v26 = insertelement <3 x double> zeroinitializer, double 2.0, i32 2
 %v27 = insertelement <4 x double> zeroinitializer, double 3.0, i32 3
 %v28 = insertelement <8 x double> zeroinitializer, double 4.0, i32 4
 %v29 = insertelement <16 x double> zeroinitializer, double 5.0, i32 7

 ret i32 0
}
