; RUN: %lli -jit-kind=mcjit -force-interpreter=true %s > /dev/null

define i32 @main() {
    zext <2 x i1> <i1 true,i1 true> to <2 x i8>
    zext <3 x i1> <i1 true,i1 true,i1 true> to <3 x i8>
    zext <2 x i1> <i1 true,i1 true> to <2 x i16>
    zext <3 x i1> <i1 true,i1 true,i1 true> to <3 x i16>
    zext <2 x i1> <i1 true,i1 true> to <2 x i32>
    zext <3 x i1> <i1 true,i1 true,i1 true> to <3 x i32>
    zext <2 x i1> <i1 true,i1 true> to <2 x i64>
    zext <3 x i1> <i1 true,i1 true,i1 true> to <3 x i64>
    zext <3 x i8> <i8 4, i8 4, i8 4> to <3 x i16>
    zext <2 x i8> <i8 -4, i8 -4> to <2 x i16>
    zext <3 x i8> <i8 4, i8 4, i8 4> to <3 x i32>
    zext <2 x i8> <i8 -4, i8 -4> to <2 x i32>
    zext <3 x i8> <i8 4, i8 4, i8 4> to <3 x i64>
    zext <2 x i8> <i8 -4, i8 -4> to <2 x i64>
    zext <3 x i16> <i16 4, i16 4, i16 4> to <3 x i32>
    zext <2 x i16> <i16 -4, i16 -4> to <2 x i32>
    zext <3 x i16> <i16 4, i16 4, i16 4> to <3 x i64>
    zext <2 x i16> <i16 -4, i16 -4> to <2 x i64>
    zext <3 x i32> <i32 4, i32 4, i32 4> to <3 x i64>
    zext <2 x i32> <i32 -4, i32 -4> to <2 x i64>


    sext <2 x i1> <i1 true,i1 true> to <2 x i8>
    sext <3 x i1> <i1 true,i1 false,i1 true> to <3 x i8>
    sext <2 x i1> <i1 true,i1 true> to <2 x i16>
    sext <3 x i1> <i1 true,i1 false,i1 true> to <3 x i16>
    sext <2 x i1> <i1 true,i1 true> to <2 x i32>
    sext <3 x i1> <i1 true,i1 false,i1 true> to <3 x i32>
    sext <2 x i1> <i1 true,i1 true> to <2 x i64>
    sext <3 x i1> <i1 true,i1 false,i1 true> to <3 x i64>
    sext <3 x i8> <i8 -4, i8 0, i8 4> to <3 x i16>
    sext <2 x i8> <i8 -4, i8 4> to <2 x i16>
    sext <3 x i8> <i8 -4, i8 0, i8 4> to <3 x i32>
    sext <2 x i8> <i8 -4, i8 4> to <2 x i32>
    sext <3 x i8> <i8 -4, i8 0, i8 4> to <3 x i64>
    sext <2 x i8> <i8 -4, i8 4> to <2 x i64>
    sext <3 x i16> <i16 -4, i16 0, i16 4> to <3 x i32>
    sext <2 x i16> <i16 -4, i16 4> to <2 x i32>
    sext <3 x i16> <i16 -4, i16 0, i16 4> to <3 x i64>
    sext <2 x i16> <i16 -4, i16 4> to <2 x i64>
    sext <3 x i32> <i32 -4, i32 0, i32 4> to <3 x i64>
    sext <2 x i32> <i32 -4, i32 4> to <2 x i64>


    uitofp <3 x i1> <i1 true,i1 false,i1 true> to <3 x float>
    uitofp <2 x i1> <i1 true,i1 true> to <2 x double>
    uitofp <3 x i8> <i8 -4,i8 0,i8 4> to <3 x float>
    uitofp <2 x i8> <i8 -4,i8 4> to <2 x double>
    uitofp <3 x i16> <i16 -4,i16 0,i16 4> to <3 x float>
    uitofp <2 x i16> <i16 -4,i16 4> to <2 x double>
    uitofp <3 x i32> <i32 -4,i32 0,i32 4> to <3 x float>
    uitofp <2 x i32> <i32 -4,i32 4> to <2 x double>
    uitofp <3 x i64> <i64 -4,i64 0,i64 4> to <3 x float>
    uitofp <2 x i64> <i64 -4,i64 4> to <2 x double>


    sitofp <3 x i1> <i1 true,i1 false,i1 true> to <3 x float>
    sitofp <2 x i1> <i1 true,i1 true> to <2 x double>
    sitofp <3 x i8> <i8 -4,i8 0,i8 4> to <3 x float>
    sitofp <2 x i8> <i8 -4,i8 4> to <2 x double>
    sitofp <3 x i16> <i16 -4,i16 0,i16 4> to <3 x float>
    sitofp <2 x i16> <i16 -4,i16 4> to <2 x double>
    sitofp <3 x i32> <i32 -4,i32 0,i32 4> to <3 x float>
    sitofp <2 x i32> <i32 -4,i32 4> to <2 x double>
    sitofp <3 x i64> <i64 -4,i64 0,i64 4> to <3 x float>
    sitofp <2 x i64> <i64 -4,i64 4> to <2 x double>

    trunc <2 x i16> <i16 -6, i16 6> to <2 x i8>
    trunc <3 x i16> <i16 -6, i16 6, i16 0> to <3 x i8>
    trunc <2 x i32> <i32 -6, i32 6> to <2 x i8>
    trunc <3 x i32> <i32 -6, i32 6, i32 0> to <3 x i8>
    trunc <2 x i32> <i32 -6, i32 6> to <2 x i16>
    trunc <3 x i32> <i32 -6, i32 6, i32 0> to <3 x i16>
    trunc <2 x i64> <i64 -6, i64 6> to <2 x i8>
    trunc <3 x i64> <i64 -6, i64 6, i64 0> to <3 x i8>
    trunc <2 x i64> <i64 -6, i64 6> to <2 x i16>
    trunc <3 x i64> <i64 -6, i64 6, i64 0> to <3 x i16>
    trunc <2 x i64> <i64 -6, i64 6> to <2 x i32>
    trunc <3 x i64> <i64 -6, i64 6, i64 0> to <3 x i32>


    fpext <2 x float>  < float 0.000000e+00, float 1.0> to <2 x double>
    fpext <3 x float>  < float 0.000000e+00, float -1.0, float 1.0> to <3 x double>

    fptosi <2 x double> < double 0.000000e+00, double 1.0> to <2 x i8>
    fptosi <3 x double> < double 0.000000e+00, double 1.0, double -1.0> to <3 x i8>
    fptosi <2 x double> < double 0.000000e+00, double 1.0> to <2 x i16>
    fptosi <3 x double> < double 0.000000e+00, double 1.0, double -1.0> to <3 x i16>
    fptosi <2 x double> < double 0.000000e+00, double 1.0> to <2 x i32>
    fptosi <3 x double> < double 0.000000e+00, double 1.0, double -1.0> to <3 x i32>
    fptosi <2 x double> < double 0.000000e+00, double 1.0> to <2 x i64>
    fptosi <3 x double> < double 0.000000e+00, double 1.0, double -1.0> to <3 x i64>

    fptoui <2 x double> < double 0.000000e+00, double 1.0> to <2 x i8>
    fptoui <3 x double> < double 0.000000e+00, double 1.0, double -1.0> to <3 x i8>
    fptoui <2 x double> < double 0.000000e+00, double 1.0> to <2 x i16>
    fptoui <3 x double> < double 0.000000e+00, double 1.0, double -1.0> to <3 x i16>
    fptoui <2 x double> < double 0.000000e+00, double 1.0> to <2 x i32>
    fptoui <3 x double> < double 0.000000e+00, double 1.0, double -1.0> to <3 x i32>
    fptoui <2 x double> < double 0.000000e+00, double 1.0> to <2 x i64>
    fptoui <3 x double> < double 0.000000e+00, double 1.0, double -1.0> to <3 x i64>

    fptrunc <2 x double> < double 0.000000e+00, double 1.0> to <2 x float>
    fptrunc <3 x double> < double 0.000000e+00, double 1.0, double -1.0> to <3 x float>

    bitcast <8 x i8> <i8 0, i8 -1, i8 2, i8 -3, i8 4, i8 -5, i8 6, i8 -7> to <4 x i16>
    bitcast <8 x i8> <i8 0, i8 -1, i8 2, i8 -3, i8 4, i8 -5, i8 6, i8 -7> to <2 x i32>
    bitcast <8 x i8> <i8 0, i8 -1, i8 2, i8 -3, i8 4, i8 -5, i8 6, i8 -7> to i64
    bitcast <8 x i8> <i8 0, i8 -1, i8 2, i8 -3, i8 4, i8 -5, i8 6, i8 -7> to <2 x float>
    bitcast <8 x i8> <i8 0, i8 -1, i8 2, i8 -3, i8 4, i8 -5, i8 6, i8 -7> to double

    bitcast <4 x i16> <i16 0, i16 -1, i16 2, i16 -3> to <8 x i8>
    bitcast <4 x i16> <i16 0, i16 -1, i16 2, i16 -3> to <2 x i32>
    bitcast <4 x i16> <i16 0, i16 -1, i16 2, i16 -3> to i64
    bitcast <4 x i16> <i16 0, i16 -1, i16 2, i16 -3> to <2 x float>
    bitcast <4 x i16> <i16 0, i16 -1, i16 2, i16 -3> to double

    bitcast <2 x i32> <i32 1, i32 -1> to <8 x i8>
    bitcast <2 x i32> <i32 1, i32 -1> to <4 x i16>
    bitcast <2 x i32> <i32 1, i32 -1> to i64
    bitcast <2 x i32> <i32 1, i32 -1> to <2 x float>
    bitcast <2 x i32> <i32 1, i32 -1> to double

    bitcast i64 1 to <8 x i8>
    bitcast i64 1 to <4 x i16>
    bitcast i64 1 to <2 x i32>
    bitcast i64 1 to <2 x float>
    bitcast i64 1 to double

    bitcast <2 x float> <float 1.0, float -1.0> to <8 x i8>
    bitcast <2 x float> <float 1.0, float -1.0> to <4 x i16>
    bitcast <2 x float> <float 1.0, float -1.0> to i64
    bitcast <2 x float> <float 1.0, float -1.0> to <2 x i32>
    bitcast <2 x float> <float 1.0, float -1.0> to double

    bitcast double 1.0 to <8 x i8>
    bitcast double 1.0 to <4 x i16>
    bitcast double 1.0 to <2 x i32>
    bitcast double 1.0 to <2 x float>
    bitcast double 1.0 to i64

    ret i32 0
}
