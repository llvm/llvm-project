; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/CoyoteBench/almabench.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/CoyoteBench/almabench.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@a = internal unnamed_addr constant [8 x [3 x double]] [[3 x double] [double 0x3FD8C637FD3B6253, double 0.000000e+00, double 0.000000e+00], [3 x double] [double 0x3FE725849423E3E0, double 0.000000e+00, double 0.000000e+00], [3 x double] [double 0x3FF000011136AEF5, double 0.000000e+00, double 0.000000e+00], [3 x double] [double 0x3FF860FD96F0D223, double 3.000000e-10, double 0.000000e+00], [3 x double] [double 0x4014CF7737365089, double 1.913200e-06, double -3.900000e-09], [3 x double] [double 0x40231C1D0EBB7C0F, double -2.138960e-05, double 4.440000e-08], [3 x double] [double 0x403337EC14C35EFA, double -3.716000e-07, double 0x3E7A47A3038502A4], [3 x double] [double 0x403E1C425059FB17, double -1.663500e-06, double 6.860000e-08]], align 8
@dlm = internal unnamed_addr constant [8 x [3 x double]] [[3 x double] [double 0x406F88076B035926, double 0x41F40BBCADEE3CB4, double -1.927890e+00], [3 x double] [double 0x4066BF5A874FEAFA, double 0x41DF6432F5157881, double 5.938100e-01], [3 x double] [double 0x40591DDA6DBF7622, double 0x41D34FC2F3B56502, double -2.044110e+00], [3 x double] [double 0x407636ED90F7B482, double 0x41C4890A4B784DFD, double 9.426400e-01], [3 x double] [double 0x40412CFE90EA1D96, double 0x419A0C7E6F1EA0BA, double 0xC03E9A915379FA98], [3 x double] [double 0x404909E9B1DFE17D, double 0x4184FA9E14756430, double 0x4052E76ED677707A], [3 x double] [double 0x4073A0E14D09C902, double 0x416D6BA57E0EFDCA, double -1.750830e+00], [3 x double] [double 0x4073059422411D82, double 0x415E0127CD46B26C, double 2.110300e-01]], align 8
@e = internal unnamed_addr constant [8 x [3 x double]] [[3 x double] [double 0x3FCA52242A37D430, double 0x3F2ABF4B9459E7F4, double -2.834900e-06], [3 x double] [double 0x3F7BBCDE77820827, double 0xBF3F4DAC25FB4BC2, double 9.812700e-06], [3 x double] [double 0x3F911C1175CC9F7B, double 0xBF3B8C8FA536F731, double -1.267340e-05], [3 x double] [double 0x3FB7E91AD74BF5B0, double 0x3F4DA66143B5E407, double -8.064100e-06], [3 x double] [double 0x3FA8D4B857E48742, double 0x3F5ABE2B9A18B7B5, double -4.713660e-05], [3 x double] [double 0x3FAC70CE5FA41E66, double 0xBF6C6594A86FD58E, double -6.436390e-05], [3 x double] [double 0x3FA7BF479022D287, double 0xBF31E2FE6AE927D8, double 7.891300e-06], [3 x double] [double 0x3F835D88E0FE76D8, double 6.032630e-05, double 0.000000e+00]], align 8
@pi = internal unnamed_addr constant [8 x [3 x double]] [[3 x double] [double 0x40535D310DE9F882, double 0x40B6571DAB9F559B, double -4.830160e+00], [3 x double] [double 0x40607209DADFB507, double 0x4065EF9096BB98C8, double 0xC07F27B59DDC1E79], [3 x double] [double 0x4059BBFD82CD2461, double 0x40C6AE2D2BD3C361, double 0x404AA34C6E6D9BE5], [3 x double] [double 0x407500F6B7DFD5BE, double 0x40CF363AC3222920, double -6.232800e+01], [3 x double] [double 0x402CA993F265B897, double 0x40BE4EC06AD2DCB1, double 0x40703F599ED7C6FC], [3 x double] [double 0x405743A9C7642D26, double 0x40D3EADFA415F45E, double 0x4067C84DFCE3150E], [3 x double] [double 0x4065A02B58283528, double 0x40A91F1FF04577D9, double 0xC0410BE37DE939EB], [3 x double] [double 0x40480F65305B6785, double 0x40906AE060FE4799, double 0x403B65ACEEE0F3CB]], align 8
@dinc = internal unnamed_addr constant [8 x [3 x double]] [[3 x double] [double 0x401C051B1D92B7FE, double 0xC06AC83387160957, double 2.897700e-01], [3 x double] [double 0x400B28447E34386C, double 0xC03ED828A1DFB939, double 0xC0275B52007DD441], [3 x double] [double 0.000000e+00, double 0x407D5F90F51AC9B0, double -3.350530e+00], [3 x double] [double 0x3FFD987ACB2252BB, double 0xC072551355475A32, double -8.118300e+00], [3 x double] [double 0x3FF4DA2E7A10E830, double 0xC051E3C504816F00, double 0x4027E7EBAF102364], [3 x double] [double 0x4003E939471E778F, double 0x4056F686594AF4F1, double 0xC031A989374BC6A8], [3 x double] [double 0x3FE8BE07677D67B5, double 0xC04E5D15DF6555C5, double 1.257590e+00], [3 x double] [double 0x3FFC51B9CE9853F4, double 0x40203F251C193B3A, double 8.135000e-02]], align 8
@omega = internal unnamed_addr constant [8 x [3 x double]] [[3 x double] [double 0x40482A5AB400A313, double 0xC0B1A3379F01B867, double 0xC03FCC8605681ECD], [3 x double] [double 0x40532B83CFF8FC2B, double 0xC0C38C3DA31A4BDC, double 0xC049A9BEF49CF56F], [3 x double] [double 0x4065DBF10E4FF9E8, double 0xC0C0F3A29A804966, double 0x402EAF0ED3D859C9], [3 x double] [double 0x4048C76F992A88EB, double 0xC0C4BE7350092CCF, double 0xC06CD25F84CAD57C], [3 x double] [double 0x40591DB8D838BBB3, double 0x40B8DA091DBCA969, double 0x4074685935FC3B4F], [3 x double] [double 0x405C6A9797E1B38F, double 0xC0C20C1986983516, double 0xC0508F320D9945B7], [3 x double] [double 0x405280619982C872, double 0x40A4DA4CF80DC337, double 0x40623E1187E7C06E], [3 x double] [double 0x40607916FEBF632D, double 0xC06BBE2EDBB59DDC, double -7.872800e-01]], align 8
@kp = internal unnamed_addr constant [8 x [9 x double]] [[9 x double] [double 6.961300e+04, double 7.564500e+04, double 8.830600e+04, double 5.989900e+04, double 1.574600e+04, double 7.108700e+04, double 1.421730e+05, double 3.086000e+03, double 0.000000e+00], [9 x double] [double 2.186300e+04, double 3.279400e+04, double 2.693400e+04, double 1.093100e+04, double 2.625000e+04, double 4.372500e+04, double 5.386700e+04, double 2.893900e+04, double 0.000000e+00], [9 x double] [double 1.600200e+04, double 2.186300e+04, double 3.200400e+04, double 1.093100e+04, double 1.452900e+04, double 1.636800e+04, double 1.531800e+04, double 3.279400e+04, double 0.000000e+00], [9 x double] [double 6.345000e+03, double 7.818000e+03, double 1.563600e+04, double 7.077000e+03, double 8.184000e+03, double 1.416300e+04, double 1.107000e+03, double 4.872000e+03, double 0.000000e+00], [9 x double] [double 1.760000e+03, double 1.454000e+03, double 1.167000e+03, double 8.800000e+02, double 2.870000e+02, double 2.640000e+03, double 1.900000e+01, double 2.047000e+03, double 1.454000e+03], [9 x double] [double 5.740000e+02, double 0.000000e+00, double 8.800000e+02, double 2.870000e+02, double 1.900000e+01, double 1.760000e+03, double 1.167000e+03, double 3.060000e+02, double 5.740000e+02], [9 x double] [double 2.040000e+02, double 0.000000e+00, double 1.770000e+02, double 1.265000e+03, double 4.000000e+00, double 3.850000e+02, double 2.000000e+02, double 2.080000e+02, double 2.040000e+02], [9 x double] [double 0.000000e+00, double 1.020000e+02, double 1.060000e+02, double 4.000000e+00, double 9.800000e+01, double 1.367000e+03, double 4.870000e+02, double 2.040000e+02, double 0.000000e+00]], align 8
@kq = internal unnamed_addr constant [8 x [10 x double]] [[10 x double] [double 3.086000e+03, double 1.574600e+04, double 6.961300e+04, double 5.989900e+04, double 7.564500e+04, double 8.830600e+04, double 1.266100e+04, double 2.658000e+03, double 0.000000e+00, double 0.000000e+00], [10 x double] [double 2.186300e+04, double 3.279400e+04, double 1.093100e+04, double 7.300000e+01, double 4.387000e+03, double 2.693400e+04, double 1.473000e+03, double 2.157000e+03, double 0.000000e+00, double 0.000000e+00], [10 x double] [double 1.000000e+01, double 1.600200e+04, double 2.186300e+04, double 1.093100e+04, double 1.473000e+03, double 3.200400e+04, double 4.387000e+03, double 7.300000e+01, double 0.000000e+00, double 0.000000e+00], [10 x double] [double 1.000000e+01, double 6.345000e+03, double 7.818000e+03, double 1.107000e+03, double 1.563600e+04, double 7.077000e+03, double 8.184000e+03, double 5.320000e+02, double 1.000000e+01, double 0.000000e+00], [10 x double] [double 1.900000e+01, double 1.760000e+03, double 1.454000e+03, double 2.870000e+02, double 1.167000e+03, double 8.800000e+02, double 5.740000e+02, double 2.640000e+03, double 1.900000e+01, double 1.454000e+03], [10 x double] [double 1.900000e+01, double 5.740000e+02, double 2.870000e+02, double 3.060000e+02, double 1.760000e+03, double 1.200000e+01, double 3.100000e+01, double 3.800000e+01, double 1.900000e+01, double 5.740000e+02], [10 x double] [double 4.000000e+00, double 2.040000e+02, double 1.770000e+02, double 8.000000e+00, double 3.100000e+01, double 2.000000e+02, double 1.265000e+03, double 1.020000e+02, double 4.000000e+00, double 2.040000e+02], [10 x double] [double 4.000000e+00, double 1.020000e+02, double 1.060000e+02, double 8.000000e+00, double 9.800000e+01, double 1.367000e+03, double 4.870000e+02, double 2.040000e+02, double 4.000000e+00, double 1.020000e+02]], align 8
@ca = internal unnamed_addr constant [8 x [9 x double]] [[9 x double] [double 4.000000e+00, double -1.300000e+01, double 1.100000e+01, double -9.000000e+00, double -9.000000e+00, double -3.000000e+00, double -1.000000e+00, double 4.000000e+00, double 0.000000e+00], [9 x double] [double -1.560000e+02, double 5.900000e+01, double -4.200000e+01, double 6.000000e+00, double 1.900000e+01, double -2.000000e+01, double -1.000000e+01, double -1.200000e+01, double 0.000000e+00], [9 x double] [double 6.400000e+01, double -1.520000e+02, double 6.200000e+01, double -8.000000e+00, double 3.200000e+01, double -4.100000e+01, double 1.900000e+01, double -1.100000e+01, double 0.000000e+00], [9 x double] [double 1.240000e+02, double 6.210000e+02, double -1.450000e+02, double 2.080000e+02, double 5.400000e+01, double -5.700000e+01, double 3.000000e+01, double 1.500000e+01, double 0.000000e+00], [9 x double] [double -2.343700e+04, double -2.634000e+03, double 6.601000e+03, double 6.259000e+03, double -1.507000e+03, double -1.821000e+03, double 2.620000e+03, double -2.115000e+03, double -1.489000e+03], [9 x double] [double 6.291100e+04, double -1.199190e+05, double 7.933600e+04, double 1.781400e+04, double -2.424100e+04, double 1.206800e+04, double 8.306000e+03, double -4.893000e+03, double 8.902000e+03], [9 x double] [double 3.890610e+05, double -2.621250e+05, double -4.408800e+04, double 8.387000e+03, double -2.297600e+04, double -2.093000e+03, double -6.150000e+02, double -9.720000e+03, double 6.633000e+03], [9 x double] [double -4.122350e+05, double -1.570460e+05, double -3.143000e+04, double 3.781700e+04, double -9.740000e+03, double -1.300000e+01, double -7.449000e+03, double 9.644000e+03, double 0.000000e+00]], align 8
@sa = internal unnamed_addr constant [8 x [9 x double]] [[9 x double] [double -2.900000e+01, double -1.000000e+00, double 9.000000e+00, double 6.000000e+00, double -6.000000e+00, double 5.000000e+00, double 4.000000e+00, double 0.000000e+00, double 0.000000e+00], [9 x double] [double -4.800000e+01, double -1.250000e+02, double -2.600000e+01, double -3.700000e+01, double 1.800000e+01, double -1.300000e+01, double -2.000000e+01, double -2.000000e+00, double 0.000000e+00], [9 x double] [double -1.500000e+02, double -4.600000e+01, double 6.800000e+01, double 5.400000e+01, double 1.400000e+01, double 2.400000e+01, double -2.800000e+01, double 2.200000e+01, double 0.000000e+00], [9 x double] [double -6.210000e+02, double 5.320000e+02, double -6.940000e+02, double -2.000000e+01, double 1.920000e+02, double -9.400000e+01, double 7.100000e+01, double -7.300000e+01, double 0.000000e+00], [9 x double] [double -1.461400e+04, double -1.982800e+04, double -5.869000e+03, double 1.881000e+03, double -4.372000e+03, double -2.255000e+03, double 7.820000e+02, double 9.300000e+02, double 9.130000e+02], [9 x double] [double 1.397370e+05, double 0.000000e+00, double 2.466700e+04, double 5.112300e+04, double -5.102000e+03, double 7.429000e+03, double -4.095000e+03, double -1.976000e+03, double -9.566000e+03], [9 x double] [double -1.380810e+05, double 0.000000e+00, double 3.720500e+04, double -4.903900e+04, double -4.190100e+04, double -3.387200e+04, double -2.703700e+04, double -1.247400e+04, double 1.879700e+04], [9 x double] [double 0.000000e+00, double 2.849200e+04, double 1.332360e+05, double 6.965400e+04, double 5.232200e+04, double -4.957700e+04, double -2.643000e+04, double -3.593000e+03, double 0.000000e+00]], align 8
@cl = internal unnamed_addr constant [8 x [10 x double]] [[10 x double] [double 2.100000e+01, double -9.500000e+01, double -1.570000e+02, double 4.100000e+01, double -5.000000e+00, double 4.200000e+01, double 2.300000e+01, double 3.000000e+01, double 0.000000e+00, double 0.000000e+00], [10 x double] [double -1.600000e+02, double -3.130000e+02, double -2.350000e+02, double 6.000000e+01, double -7.400000e+01, double -7.600000e+01, double -2.700000e+01, double 3.400000e+01, double 0.000000e+00, double 0.000000e+00], [10 x double] [double -3.250000e+02, double -3.220000e+02, double -7.900000e+01, double 2.320000e+02, double -5.200000e+01, double 9.700000e+01, double 5.500000e+01, double -4.100000e+01, double 0.000000e+00, double 0.000000e+00], [10 x double] [double 2.268000e+03, double -9.790000e+02, double 8.020000e+02, double 6.020000e+02, double -6.680000e+02, double -3.300000e+01, double 3.450000e+02, double 2.010000e+02, double -5.500000e+01, double 0.000000e+00], [10 x double] [double 7.610000e+03, double -4.997000e+03, double -7.689000e+03, double -5.841000e+03, double -2.617000e+03, double 1.115000e+03, double -7.480000e+02, double -6.070000e+02, double 6.074000e+03, double 3.540000e+02], [10 x double] [double -1.854900e+04, double 3.012500e+04, double 2.001200e+04, double -7.300000e+02, double 8.240000e+02, double 2.300000e+01, double 1.289000e+03, double -3.520000e+02, double -1.476700e+04, double -2.062000e+03], [10 x double] [double -1.352450e+05, double -1.459400e+04, double 4.197000e+03, double -4.030000e+03, double -5.630000e+03, double -2.898000e+03, double 2.540000e+03, double -3.060000e+02, double 2.939000e+03, double 1.986000e+03], [10 x double] [double 8.994800e+04, double 2.103000e+03, double 8.963000e+03, double 2.695000e+03, double 3.682000e+03, double 1.648000e+03, double 8.660000e+02, double -1.540000e+02, double -1.963000e+03, double -2.830000e+02]], align 8
@sl = internal unnamed_addr constant [8 x [10 x double]] [[10 x double] [double -3.420000e+02, double 1.360000e+02, double -2.300000e+01, double 6.200000e+01, double 6.600000e+01, double -5.200000e+01, double -3.300000e+01, double 1.700000e+01, double 0.000000e+00, double 0.000000e+00], [10 x double] [double 5.240000e+02, double -1.490000e+02, double -3.500000e+01, double 1.170000e+02, double 1.510000e+02, double 1.220000e+02, double -7.100000e+01, double -6.200000e+01, double 0.000000e+00, double 0.000000e+00], [10 x double] [double -1.050000e+02, double -1.370000e+02, double 2.580000e+02, double 3.500000e+01, double -1.160000e+02, double -8.800000e+01, double -1.120000e+02, double -8.000000e+01, double 0.000000e+00, double 0.000000e+00], [10 x double] [double 8.540000e+02, double -2.050000e+02, double -9.360000e+02, double -2.400000e+02, double 1.400000e+02, double -3.410000e+02, double -9.700000e+01, double -2.320000e+02, double 5.360000e+02, double 0.000000e+00], [10 x double] [double -5.698000e+04, double 8.016000e+03, double 1.012000e+03, double 1.448000e+03, double -3.024000e+03, double -3.710000e+03, double 3.180000e+02, double 5.030000e+02, double 3.767000e+03, double 5.770000e+02], [10 x double] [double 1.386060e+05, double -1.347800e+04, double -4.964000e+03, double 1.441000e+03, double -1.319000e+03, double -1.482000e+03, double 4.270000e+02, double 1.236000e+03, double -9.167000e+03, double -1.918000e+03], [10 x double] [double 7.123400e+04, double -4.111600e+04, double 5.334000e+03, double -4.935000e+03, double -1.848000e+03, double 6.600000e+01, double 4.340000e+02, double -1.748000e+03, double 3.780000e+03, double -7.010000e+02], [10 x double] [double -4.764500e+04, double 1.164700e+04, double 2.166000e+03, double 3.194000e+03, double 6.790000e+02, double 0.000000e+00, double -2.440000e+02, double -4.190000e+02, double -2.531000e+03, double 4.800000e+01]], align 8
@amas = internal unnamed_addr constant [8 x double] [double 6.023600e+06, double 0x4118EF2E00000000, double 0x4114131200000000, double 3.098710e+06, double 0x40905D6B851EB852, double 3.498500e+03, double 2.286900e+04, double 1.931400e+04], align 8
@.str.1 = private unnamed_addr constant [10 x i8] c"%f %f %f\0A\00", align 1
@stdout = external local_unnamed_addr global ptr, align 8

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(errnomem: write) uwtable
define dso_local double @anpm(double noundef %0) local_unnamed_addr #0 {
  %2 = tail call double @fmod(double noundef %0, double noundef 0x401921FB54442D18) #8, !tbaa !6
  %3 = tail call double @llvm.fabs.f64(double %2)
  %4 = fcmp ult double %3, 0x400921FB54442D18
  %5 = fcmp olt double %0, 0.000000e+00
  %6 = select i1 %5, double 0xC01921FB54442D18, double 0x401921FB54442D18
  %7 = fsub double %2, %6
  %8 = select i1 %4, double %2, double %7
  ret double %8
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare double @fmod(double noundef, double noundef) local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fabs.f64(double) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, errnomem: write) uwtable
define dso_local void @planetpv(ptr noundef readonly captures(none) %0, i32 noundef %1, ptr noundef writeonly captures(none) initializes((0, 48)) %2) local_unnamed_addr #4 {
  %4 = load double, ptr %0, align 8, !tbaa !10
  %5 = fadd double %4, 0xC142B42C80000000
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %7 = load double, ptr %6, align 8, !tbaa !10
  %8 = fadd double %5, %7
  %9 = fdiv double %8, 3.652500e+05
  %10 = sext i32 %1 to i64
  %11 = getelementptr inbounds [3 x double], ptr @a, i64 %10
  %12 = load double, ptr %11, align 8, !tbaa !10
  %13 = getelementptr inbounds nuw i8, ptr %11, i64 8
  %14 = load double, ptr %13, align 8, !tbaa !10
  %15 = getelementptr inbounds nuw i8, ptr %11, i64 16
  %16 = load double, ptr %15, align 8, !tbaa !10
  %17 = getelementptr inbounds [3 x double], ptr @dlm, i64 %10
  %18 = load double, ptr %17, align 8, !tbaa !10
  %19 = getelementptr inbounds nuw i8, ptr %17, i64 8
  %20 = load double, ptr %19, align 8, !tbaa !10
  %21 = getelementptr inbounds nuw i8, ptr %17, i64 16
  %22 = load double, ptr %21, align 8, !tbaa !10
  %23 = tail call double @llvm.fmuladd.f64(double %22, double %9, double %20)
  %24 = fmul double %9, %23
  %25 = tail call double @llvm.fmuladd.f64(double %18, double 3.600000e+03, double %24)
  %26 = fmul double %25, 0x3ED455A5B2FF8F9D
  %27 = getelementptr inbounds [3 x double], ptr @e, i64 %10
  %28 = load double, ptr %27, align 8, !tbaa !10
  %29 = getelementptr inbounds nuw i8, ptr %27, i64 8
  %30 = load double, ptr %29, align 8, !tbaa !10
  %31 = getelementptr inbounds nuw i8, ptr %27, i64 16
  %32 = load double, ptr %31, align 8, !tbaa !10
  %33 = getelementptr inbounds [3 x double], ptr @pi, i64 %10
  %34 = load double, ptr %33, align 8, !tbaa !10
  %35 = getelementptr inbounds nuw i8, ptr %33, i64 8
  %36 = load double, ptr %35, align 8, !tbaa !10
  %37 = getelementptr inbounds nuw i8, ptr %33, i64 16
  %38 = load double, ptr %37, align 8, !tbaa !10
  %39 = tail call double @llvm.fmuladd.f64(double %38, double %9, double %36)
  %40 = fmul double %9, %39
  %41 = tail call double @llvm.fmuladd.f64(double %34, double 3.600000e+03, double %40)
  %42 = fmul double %41, 0x3ED455A5B2FF8F9D
  %43 = tail call double @fmod(double noundef %42, double noundef 0x401921FB54442D18) #8, !tbaa !6
  %44 = getelementptr inbounds [3 x double], ptr @dinc, i64 %10
  %45 = load double, ptr %44, align 8, !tbaa !10
  %46 = getelementptr inbounds nuw i8, ptr %44, i64 8
  %47 = load double, ptr %46, align 8, !tbaa !10
  %48 = getelementptr inbounds nuw i8, ptr %44, i64 16
  %49 = load double, ptr %48, align 8, !tbaa !10
  %50 = getelementptr inbounds [3 x double], ptr @omega, i64 %10
  %51 = load double, ptr %50, align 8, !tbaa !10
  %52 = getelementptr inbounds nuw i8, ptr %50, i64 8
  %53 = load double, ptr %52, align 8, !tbaa !10
  %54 = getelementptr inbounds nuw i8, ptr %50, i64 16
  %55 = load double, ptr %54, align 8, !tbaa !10
  %56 = tail call double @llvm.fmuladd.f64(double %55, double %9, double %53)
  %57 = fmul double %9, %56
  %58 = tail call double @llvm.fmuladd.f64(double %51, double 3.600000e+03, double %57)
  %59 = fmul double %58, 0x3ED455A5B2FF8F9D
  %60 = tail call double @fmod(double noundef %59, double noundef 0x401921FB54442D18) #8, !tbaa !6
  %61 = fmul double %9, 0x3FD702A41F2E9970
  %62 = getelementptr inbounds [9 x double], ptr @kp, i64 %10
  %63 = getelementptr inbounds [10 x double], ptr @kq, i64 %10
  %64 = getelementptr inbounds [9 x double], ptr @ca, i64 %10
  %65 = getelementptr inbounds [9 x double], ptr @sa, i64 %10
  %66 = getelementptr inbounds [10 x double], ptr @cl, i64 %10
  %67 = getelementptr inbounds [10 x double], ptr @sl, i64 %10
  %68 = load double, ptr %62, align 8, !tbaa !10
  %69 = fmul double %61, %68
  %70 = load double, ptr %63, align 8, !tbaa !10
  %71 = fmul double %61, %70
  %72 = load double, ptr %64, align 8, !tbaa !10
  %73 = tail call double @cos(double noundef %69) #8, !tbaa !6
  %74 = load double, ptr %65, align 8, !tbaa !10
  %75 = tail call double @sin(double noundef %69) #8, !tbaa !6
  %76 = load double, ptr %66, align 8, !tbaa !10
  %77 = tail call double @cos(double noundef %71) #8, !tbaa !6
  %78 = load double, ptr %67, align 8, !tbaa !10
  %79 = tail call double @sin(double noundef %71) #8, !tbaa !6
  %80 = fmul double %78, %79
  %81 = tail call double @llvm.fmuladd.f64(double %76, double %77, double %80)
  %82 = tail call double @llvm.fmuladd.f64(double %81, double 0x3E7AD7F29ABCAF48, double %26)
  %83 = getelementptr inbounds nuw i8, ptr %62, i64 8
  %84 = load double, ptr %83, align 8, !tbaa !10
  %85 = fmul double %61, %84
  %86 = getelementptr inbounds nuw i8, ptr %63, i64 8
  %87 = load double, ptr %86, align 8, !tbaa !10
  %88 = fmul double %61, %87
  %89 = getelementptr inbounds nuw i8, ptr %64, i64 8
  %90 = load double, ptr %89, align 8, !tbaa !10
  %91 = tail call double @cos(double noundef %85) #8, !tbaa !6
  %92 = getelementptr inbounds nuw i8, ptr %65, i64 8
  %93 = load double, ptr %92, align 8, !tbaa !10
  %94 = tail call double @sin(double noundef %85) #8, !tbaa !6
  %95 = getelementptr inbounds nuw i8, ptr %66, i64 8
  %96 = load double, ptr %95, align 8, !tbaa !10
  %97 = tail call double @cos(double noundef %88) #8, !tbaa !6
  %98 = getelementptr inbounds nuw i8, ptr %67, i64 8
  %99 = load double, ptr %98, align 8, !tbaa !10
  %100 = tail call double @sin(double noundef %88) #8, !tbaa !6
  %101 = fmul double %99, %100
  %102 = tail call double @llvm.fmuladd.f64(double %96, double %97, double %101)
  %103 = tail call double @llvm.fmuladd.f64(double %102, double 0x3E7AD7F29ABCAF48, double %82)
  %104 = getelementptr inbounds nuw i8, ptr %62, i64 16
  %105 = load double, ptr %104, align 8, !tbaa !10
  %106 = fmul double %61, %105
  %107 = getelementptr inbounds nuw i8, ptr %63, i64 16
  %108 = load double, ptr %107, align 8, !tbaa !10
  %109 = fmul double %61, %108
  %110 = getelementptr inbounds nuw i8, ptr %64, i64 16
  %111 = load double, ptr %110, align 8, !tbaa !10
  %112 = tail call double @cos(double noundef %106) #8, !tbaa !6
  %113 = getelementptr inbounds nuw i8, ptr %65, i64 16
  %114 = load double, ptr %113, align 8, !tbaa !10
  %115 = tail call double @sin(double noundef %106) #8, !tbaa !6
  %116 = getelementptr inbounds nuw i8, ptr %66, i64 16
  %117 = load double, ptr %116, align 8, !tbaa !10
  %118 = tail call double @cos(double noundef %109) #8, !tbaa !6
  %119 = getelementptr inbounds nuw i8, ptr %67, i64 16
  %120 = load double, ptr %119, align 8, !tbaa !10
  %121 = tail call double @sin(double noundef %109) #8, !tbaa !6
  %122 = fmul double %120, %121
  %123 = tail call double @llvm.fmuladd.f64(double %117, double %118, double %122)
  %124 = tail call double @llvm.fmuladd.f64(double %123, double 0x3E7AD7F29ABCAF48, double %103)
  %125 = getelementptr inbounds nuw i8, ptr %62, i64 24
  %126 = load double, ptr %125, align 8, !tbaa !10
  %127 = fmul double %61, %126
  %128 = getelementptr inbounds nuw i8, ptr %63, i64 24
  %129 = load double, ptr %128, align 8, !tbaa !10
  %130 = fmul double %61, %129
  %131 = getelementptr inbounds nuw i8, ptr %64, i64 24
  %132 = load double, ptr %131, align 8, !tbaa !10
  %133 = tail call double @cos(double noundef %127) #8, !tbaa !6
  %134 = getelementptr inbounds nuw i8, ptr %65, i64 24
  %135 = load double, ptr %134, align 8, !tbaa !10
  %136 = tail call double @sin(double noundef %127) #8, !tbaa !6
  %137 = getelementptr inbounds nuw i8, ptr %66, i64 24
  %138 = load double, ptr %137, align 8, !tbaa !10
  %139 = tail call double @cos(double noundef %130) #8, !tbaa !6
  %140 = getelementptr inbounds nuw i8, ptr %67, i64 24
  %141 = load double, ptr %140, align 8, !tbaa !10
  %142 = tail call double @sin(double noundef %130) #8, !tbaa !6
  %143 = fmul double %141, %142
  %144 = tail call double @llvm.fmuladd.f64(double %138, double %139, double %143)
  %145 = tail call double @llvm.fmuladd.f64(double %144, double 0x3E7AD7F29ABCAF48, double %124)
  %146 = getelementptr inbounds nuw i8, ptr %62, i64 32
  %147 = load double, ptr %146, align 8, !tbaa !10
  %148 = fmul double %61, %147
  %149 = getelementptr inbounds nuw i8, ptr %63, i64 32
  %150 = load double, ptr %149, align 8, !tbaa !10
  %151 = fmul double %61, %150
  %152 = getelementptr inbounds nuw i8, ptr %64, i64 32
  %153 = load double, ptr %152, align 8, !tbaa !10
  %154 = tail call double @cos(double noundef %148) #8, !tbaa !6
  %155 = getelementptr inbounds nuw i8, ptr %65, i64 32
  %156 = load double, ptr %155, align 8, !tbaa !10
  %157 = tail call double @sin(double noundef %148) #8, !tbaa !6
  %158 = getelementptr inbounds nuw i8, ptr %66, i64 32
  %159 = load double, ptr %158, align 8, !tbaa !10
  %160 = tail call double @cos(double noundef %151) #8, !tbaa !6
  %161 = getelementptr inbounds nuw i8, ptr %67, i64 32
  %162 = load double, ptr %161, align 8, !tbaa !10
  %163 = tail call double @sin(double noundef %151) #8, !tbaa !6
  %164 = fmul double %162, %163
  %165 = tail call double @llvm.fmuladd.f64(double %159, double %160, double %164)
  %166 = tail call double @llvm.fmuladd.f64(double %165, double 0x3E7AD7F29ABCAF48, double %145)
  %167 = getelementptr inbounds nuw i8, ptr %62, i64 40
  %168 = load double, ptr %167, align 8, !tbaa !10
  %169 = fmul double %61, %168
  %170 = getelementptr inbounds nuw i8, ptr %63, i64 40
  %171 = load double, ptr %170, align 8, !tbaa !10
  %172 = fmul double %61, %171
  %173 = getelementptr inbounds nuw i8, ptr %64, i64 40
  %174 = load double, ptr %173, align 8, !tbaa !10
  %175 = tail call double @cos(double noundef %169) #8, !tbaa !6
  %176 = getelementptr inbounds nuw i8, ptr %65, i64 40
  %177 = load double, ptr %176, align 8, !tbaa !10
  %178 = tail call double @sin(double noundef %169) #8, !tbaa !6
  %179 = getelementptr inbounds nuw i8, ptr %66, i64 40
  %180 = load double, ptr %179, align 8, !tbaa !10
  %181 = tail call double @cos(double noundef %172) #8, !tbaa !6
  %182 = getelementptr inbounds nuw i8, ptr %67, i64 40
  %183 = load double, ptr %182, align 8, !tbaa !10
  %184 = tail call double @sin(double noundef %172) #8, !tbaa !6
  %185 = fmul double %183, %184
  %186 = tail call double @llvm.fmuladd.f64(double %180, double %181, double %185)
  %187 = tail call double @llvm.fmuladd.f64(double %186, double 0x3E7AD7F29ABCAF48, double %166)
  %188 = getelementptr inbounds nuw i8, ptr %62, i64 48
  %189 = load double, ptr %188, align 8, !tbaa !10
  %190 = fmul double %61, %189
  %191 = getelementptr inbounds nuw i8, ptr %63, i64 48
  %192 = load double, ptr %191, align 8, !tbaa !10
  %193 = fmul double %61, %192
  %194 = getelementptr inbounds nuw i8, ptr %64, i64 48
  %195 = load double, ptr %194, align 8, !tbaa !10
  %196 = tail call double @cos(double noundef %190) #8, !tbaa !6
  %197 = getelementptr inbounds nuw i8, ptr %65, i64 48
  %198 = load double, ptr %197, align 8, !tbaa !10
  %199 = tail call double @sin(double noundef %190) #8, !tbaa !6
  %200 = getelementptr inbounds nuw i8, ptr %66, i64 48
  %201 = load double, ptr %200, align 8, !tbaa !10
  %202 = tail call double @cos(double noundef %193) #8, !tbaa !6
  %203 = getelementptr inbounds nuw i8, ptr %67, i64 48
  %204 = load double, ptr %203, align 8, !tbaa !10
  %205 = tail call double @sin(double noundef %193) #8, !tbaa !6
  %206 = fmul double %204, %205
  %207 = tail call double @llvm.fmuladd.f64(double %201, double %202, double %206)
  %208 = tail call double @llvm.fmuladd.f64(double %207, double 0x3E7AD7F29ABCAF48, double %187)
  %209 = getelementptr inbounds nuw i8, ptr %62, i64 56
  %210 = load double, ptr %209, align 8, !tbaa !10
  %211 = fmul double %61, %210
  %212 = getelementptr inbounds nuw i8, ptr %63, i64 56
  %213 = load double, ptr %212, align 8, !tbaa !10
  %214 = fmul double %61, %213
  %215 = getelementptr inbounds nuw i8, ptr %64, i64 56
  %216 = load double, ptr %215, align 8, !tbaa !10
  %217 = tail call double @cos(double noundef %211) #8, !tbaa !6
  %218 = getelementptr inbounds nuw i8, ptr %65, i64 56
  %219 = load double, ptr %218, align 8, !tbaa !10
  %220 = tail call double @sin(double noundef %211) #8, !tbaa !6
  %221 = getelementptr inbounds nuw i8, ptr %66, i64 56
  %222 = load double, ptr %221, align 8, !tbaa !10
  %223 = tail call double @cos(double noundef %214) #8, !tbaa !6
  %224 = getelementptr inbounds nuw i8, ptr %67, i64 56
  %225 = load double, ptr %224, align 8, !tbaa !10
  %226 = tail call double @sin(double noundef %214) #8, !tbaa !6
  %227 = fmul double %225, %226
  %228 = tail call double @llvm.fmuladd.f64(double %222, double %223, double %227)
  %229 = tail call double @llvm.fmuladd.f64(double %228, double 0x3E7AD7F29ABCAF48, double %208)
  %230 = getelementptr inbounds [9 x double], ptr @kp, i64 %10, i64 8
  %231 = load double, ptr %230, align 8, !tbaa !10
  %232 = fmul double %61, %231
  %233 = getelementptr inbounds [9 x double], ptr @ca, i64 %10, i64 8
  %234 = load double, ptr %233, align 8, !tbaa !10
  %235 = tail call double @cos(double noundef %232) #8, !tbaa !6
  %236 = getelementptr inbounds [9 x double], ptr @sa, i64 %10, i64 8
  %237 = load double, ptr %236, align 8, !tbaa !10
  %238 = tail call double @sin(double noundef %232) #8, !tbaa !6
  %239 = getelementptr inbounds nuw i8, ptr %63, i64 64
  %240 = load double, ptr %239, align 8, !tbaa !10
  %241 = fmul double %61, %240
  %242 = getelementptr inbounds nuw i8, ptr %66, i64 64
  %243 = load double, ptr %242, align 8, !tbaa !10
  %244 = tail call double @cos(double noundef %241) #8, !tbaa !6
  %245 = getelementptr inbounds nuw i8, ptr %67, i64 64
  %246 = load double, ptr %245, align 8, !tbaa !10
  %247 = tail call double @sin(double noundef %241) #8, !tbaa !6
  %248 = fmul double %246, %247
  %249 = tail call double @llvm.fmuladd.f64(double %243, double %244, double %248)
  %250 = fmul double %9, %249
  %251 = tail call double @llvm.fmuladd.f64(double %250, double 0x3E7AD7F29ABCAF48, double %229)
  %252 = getelementptr inbounds nuw i8, ptr %63, i64 72
  %253 = load double, ptr %252, align 8, !tbaa !10
  %254 = fmul double %61, %253
  %255 = getelementptr inbounds nuw i8, ptr %66, i64 72
  %256 = load double, ptr %255, align 8, !tbaa !10
  %257 = tail call double @cos(double noundef %254) #8, !tbaa !6
  %258 = getelementptr inbounds nuw i8, ptr %67, i64 72
  %259 = load double, ptr %258, align 8, !tbaa !10
  %260 = tail call double @sin(double noundef %254) #8, !tbaa !6
  %261 = fmul double %259, %260
  %262 = tail call double @llvm.fmuladd.f64(double %256, double %257, double %261)
  %263 = fmul double %9, %262
  %264 = tail call double @llvm.fmuladd.f64(double %263, double 0x3E7AD7F29ABCAF48, double %251)
  %265 = tail call double @llvm.fmuladd.f64(double %32, double %9, double %30)
  %266 = tail call double @llvm.fmuladd.f64(double %265, double %9, double %28)
  %267 = tail call double @llvm.fabs.f64(double %43)
  %268 = fcmp ult double %267, 0x400921FB54442D18
  %269 = fcmp olt double %42, 0.000000e+00
  %270 = select i1 %269, double 0xC01921FB54442D18, double 0x401921FB54442D18
  %271 = fsub double %43, %270
  %272 = select i1 %268, double %43, double %271
  %273 = tail call double @fmod(double noundef %264, double noundef 0x401921FB54442D18) #8, !tbaa !6
  %274 = fsub double %273, %272
  %275 = tail call double @sin(double noundef %274) #8, !tbaa !6
  %276 = tail call double @llvm.fmuladd.f64(double %266, double %275, double %274)
  %277 = fneg double %266
  %278 = fsub double %274, %276
  %279 = tail call double @sin(double noundef %276) #8, !tbaa !6
  %280 = tail call double @llvm.fmuladd.f64(double %266, double %279, double %278)
  %281 = tail call double @cos(double noundef %276) #8, !tbaa !6
  %282 = tail call double @llvm.fmuladd.f64(double %277, double %281, double 1.000000e+00)
  %283 = fdiv double %280, %282
  %284 = fadd double %276, %283
  %285 = tail call double @llvm.fabs.f64(double %283)
  %286 = fcmp olt double %285, 0x3D719799812DEA11
  br i1 %286, label %375, label %287

287:                                              ; preds = %3
  %288 = fsub double %274, %284
  %289 = tail call double @sin(double noundef %284) #8, !tbaa !6
  %290 = tail call double @llvm.fmuladd.f64(double %266, double %289, double %288)
  %291 = tail call double @cos(double noundef %284) #8, !tbaa !6
  %292 = tail call double @llvm.fmuladd.f64(double %277, double %291, double 1.000000e+00)
  %293 = fdiv double %290, %292
  %294 = fadd double %284, %293
  %295 = tail call double @llvm.fabs.f64(double %293)
  %296 = fcmp olt double %295, 0x3D719799812DEA11
  br i1 %296, label %375, label %297

297:                                              ; preds = %287
  %298 = fsub double %274, %294
  %299 = tail call double @sin(double noundef %294) #8, !tbaa !6
  %300 = tail call double @llvm.fmuladd.f64(double %266, double %299, double %298)
  %301 = tail call double @cos(double noundef %294) #8, !tbaa !6
  %302 = tail call double @llvm.fmuladd.f64(double %277, double %301, double 1.000000e+00)
  %303 = fdiv double %300, %302
  %304 = fadd double %294, %303
  %305 = tail call double @llvm.fabs.f64(double %303)
  %306 = fcmp olt double %305, 0x3D719799812DEA11
  br i1 %306, label %375, label %307

307:                                              ; preds = %297
  %308 = fsub double %274, %304
  %309 = tail call double @sin(double noundef %304) #8, !tbaa !6
  %310 = tail call double @llvm.fmuladd.f64(double %266, double %309, double %308)
  %311 = tail call double @cos(double noundef %304) #8, !tbaa !6
  %312 = tail call double @llvm.fmuladd.f64(double %277, double %311, double 1.000000e+00)
  %313 = fdiv double %310, %312
  %314 = fadd double %304, %313
  %315 = tail call double @llvm.fabs.f64(double %313)
  %316 = fcmp olt double %315, 0x3D719799812DEA11
  br i1 %316, label %375, label %317

317:                                              ; preds = %307
  %318 = fsub double %274, %314
  %319 = tail call double @sin(double noundef %314) #8, !tbaa !6
  %320 = tail call double @llvm.fmuladd.f64(double %266, double %319, double %318)
  %321 = tail call double @cos(double noundef %314) #8, !tbaa !6
  %322 = tail call double @llvm.fmuladd.f64(double %277, double %321, double 1.000000e+00)
  %323 = fdiv double %320, %322
  %324 = fadd double %314, %323
  %325 = tail call double @llvm.fabs.f64(double %323)
  %326 = fcmp olt double %325, 0x3D719799812DEA11
  br i1 %326, label %375, label %327

327:                                              ; preds = %317
  %328 = fsub double %274, %324
  %329 = tail call double @sin(double noundef %324) #8, !tbaa !6
  %330 = tail call double @llvm.fmuladd.f64(double %266, double %329, double %328)
  %331 = tail call double @cos(double noundef %324) #8, !tbaa !6
  %332 = tail call double @llvm.fmuladd.f64(double %277, double %331, double 1.000000e+00)
  %333 = fdiv double %330, %332
  %334 = fadd double %324, %333
  %335 = tail call double @llvm.fabs.f64(double %333)
  %336 = fcmp olt double %335, 0x3D719799812DEA11
  br i1 %336, label %375, label %337

337:                                              ; preds = %327
  %338 = fsub double %274, %334
  %339 = tail call double @sin(double noundef %334) #8, !tbaa !6
  %340 = tail call double @llvm.fmuladd.f64(double %266, double %339, double %338)
  %341 = tail call double @cos(double noundef %334) #8, !tbaa !6
  %342 = tail call double @llvm.fmuladd.f64(double %277, double %341, double 1.000000e+00)
  %343 = fdiv double %340, %342
  %344 = fadd double %334, %343
  %345 = tail call double @llvm.fabs.f64(double %343)
  %346 = fcmp olt double %345, 0x3D719799812DEA11
  br i1 %346, label %375, label %347

347:                                              ; preds = %337
  %348 = fsub double %274, %344
  %349 = tail call double @sin(double noundef %344) #8, !tbaa !6
  %350 = tail call double @llvm.fmuladd.f64(double %266, double %349, double %348)
  %351 = tail call double @cos(double noundef %344) #8, !tbaa !6
  %352 = tail call double @llvm.fmuladd.f64(double %277, double %351, double 1.000000e+00)
  %353 = fdiv double %350, %352
  %354 = fadd double %344, %353
  %355 = tail call double @llvm.fabs.f64(double %353)
  %356 = fcmp olt double %355, 0x3D719799812DEA11
  br i1 %356, label %375, label %357

357:                                              ; preds = %347
  %358 = fsub double %274, %354
  %359 = tail call double @sin(double noundef %354) #8, !tbaa !6
  %360 = tail call double @llvm.fmuladd.f64(double %266, double %359, double %358)
  %361 = tail call double @cos(double noundef %354) #8, !tbaa !6
  %362 = tail call double @llvm.fmuladd.f64(double %277, double %361, double 1.000000e+00)
  %363 = fdiv double %360, %362
  %364 = fadd double %354, %363
  %365 = tail call double @llvm.fabs.f64(double %363)
  %366 = fcmp olt double %365, 0x3D719799812DEA11
  br i1 %366, label %375, label %367

367:                                              ; preds = %357
  %368 = fsub double %274, %364
  %369 = tail call double @sin(double noundef %364) #8, !tbaa !6
  %370 = tail call double @llvm.fmuladd.f64(double %266, double %369, double %368)
  %371 = tail call double @cos(double noundef %364) #8, !tbaa !6
  %372 = tail call double @llvm.fmuladd.f64(double %277, double %371, double 1.000000e+00)
  %373 = fdiv double %370, %372
  %374 = fadd double %364, %373
  br label %375

375:                                              ; preds = %367, %357, %347, %337, %327, %317, %307, %297, %287, %3
  %376 = phi double [ %284, %3 ], [ %294, %287 ], [ %304, %297 ], [ %314, %307 ], [ %324, %317 ], [ %334, %327 ], [ %344, %337 ], [ %354, %347 ], [ %364, %357 ], [ %374, %367 ]
  %377 = fmul double %219, %220
  %378 = tail call double @llvm.fmuladd.f64(double %216, double %217, double %377)
  %379 = fmul double %198, %199
  %380 = tail call double @llvm.fmuladd.f64(double %195, double %196, double %379)
  %381 = fmul double %177, %178
  %382 = tail call double @llvm.fmuladd.f64(double %174, double %175, double %381)
  %383 = fmul double %156, %157
  %384 = tail call double @llvm.fmuladd.f64(double %153, double %154, double %383)
  %385 = fmul double %135, %136
  %386 = tail call double @llvm.fmuladd.f64(double %132, double %133, double %385)
  %387 = fmul double %114, %115
  %388 = tail call double @llvm.fmuladd.f64(double %111, double %112, double %387)
  %389 = fmul double %93, %94
  %390 = tail call double @llvm.fmuladd.f64(double %90, double %91, double %389)
  %391 = fmul double %74, %75
  %392 = tail call double @llvm.fmuladd.f64(double %72, double %73, double %391)
  %393 = tail call double @llvm.fmuladd.f64(double %16, double %9, double %14)
  %394 = tail call double @llvm.fmuladd.f64(double %393, double %9, double %12)
  %395 = tail call double @llvm.fmuladd.f64(double %392, double 0x3E7AD7F29ABCAF48, double %394)
  %396 = tail call double @llvm.fmuladd.f64(double %390, double 0x3E7AD7F29ABCAF48, double %395)
  %397 = tail call double @llvm.fmuladd.f64(double %388, double 0x3E7AD7F29ABCAF48, double %396)
  %398 = tail call double @llvm.fmuladd.f64(double %386, double 0x3E7AD7F29ABCAF48, double %397)
  %399 = tail call double @llvm.fmuladd.f64(double %384, double 0x3E7AD7F29ABCAF48, double %398)
  %400 = tail call double @llvm.fmuladd.f64(double %382, double 0x3E7AD7F29ABCAF48, double %399)
  %401 = tail call double @llvm.fmuladd.f64(double %380, double 0x3E7AD7F29ABCAF48, double %400)
  %402 = tail call double @llvm.fmuladd.f64(double %378, double 0x3E7AD7F29ABCAF48, double %401)
  %403 = tail call double @llvm.fabs.f64(double %60)
  %404 = fcmp ult double %403, 0x400921FB54442D18
  %405 = fcmp olt double %59, 0.000000e+00
  %406 = select i1 %405, double 0xC01921FB54442D18, double 0x401921FB54442D18
  %407 = fsub double %60, %406
  %408 = select i1 %404, double %60, double %407
  %409 = fmul double %237, %238
  %410 = tail call double @llvm.fmuladd.f64(double %234, double %235, double %409)
  %411 = fmul double %9, %410
  %412 = tail call double @llvm.fmuladd.f64(double %411, double 0x3E7AD7F29ABCAF48, double %402)
  %413 = tail call double @llvm.fmuladd.f64(double %49, double %9, double %47)
  %414 = fmul double %9, %413
  %415 = tail call double @llvm.fmuladd.f64(double %45, double 3.600000e+03, double %414)
  %416 = fmul double %415, 0x3ED455A5B2FF8F9D
  %417 = fmul double %376, 5.000000e-01
  %418 = fadd double %266, 1.000000e+00
  %419 = fsub double 1.000000e+00, %266
  %420 = fdiv double %418, %419
  %421 = tail call double @sqrt(double noundef %420) #8, !tbaa !6
  %422 = tail call double @sin(double noundef %417) #8, !tbaa !6
  %423 = fmul double %421, %422
  %424 = tail call double @cos(double noundef %417) #8, !tbaa !6
  %425 = tail call double @atan2(double noundef %423, double noundef %424) #8, !tbaa !6
  %426 = fmul double %425, 2.000000e+00
  %427 = tail call double @cos(double noundef %376) #8, !tbaa !6
  %428 = tail call double @llvm.fmuladd.f64(double %277, double %427, double 1.000000e+00)
  %429 = fmul double %412, %428
  %430 = getelementptr inbounds double, ptr @amas, i64 %10
  %431 = load double, ptr %430, align 8, !tbaa !10
  %432 = fdiv double 1.000000e+00, %431
  %433 = fadd double %432, 1.000000e+00
  %434 = fmul double %412, %412
  %435 = fmul double %412, %434
  %436 = fdiv double %433, %435
  %437 = tail call double @sqrt(double noundef %436) #8, !tbaa !6
  %438 = fmul double %437, 0x3F919D6D51A6B69A
  %439 = fmul double %416, 5.000000e-01
  %440 = tail call double @sin(double noundef %439) #8, !tbaa !6
  %441 = tail call double @cos(double noundef %408) #8, !tbaa !6
  %442 = fmul double %440, %441
  %443 = tail call double @sin(double noundef %408) #8, !tbaa !6
  %444 = fmul double %440, %443
  %445 = fadd double %272, %426
  %446 = tail call double @sin(double noundef %445) #8, !tbaa !6
  %447 = tail call double @cos(double noundef %445) #8, !tbaa !6
  %448 = fneg double %446
  %449 = fmul double %442, %448
  %450 = tail call double @llvm.fmuladd.f64(double %444, double %447, double %449)
  %451 = fmul double %450, 2.000000e+00
  %452 = tail call double @llvm.fmuladd.f64(double %277, double %266, double 1.000000e+00)
  %453 = tail call double @sqrt(double noundef %452) #8, !tbaa !6
  %454 = fdiv double %412, %453
  %455 = tail call double @cos(double noundef %439) #8, !tbaa !6
  %456 = tail call double @sin(double noundef %272) #8, !tbaa !6
  %457 = tail call double @llvm.fmuladd.f64(double %266, double %456, double %446)
  %458 = fmul double %454, %457
  %459 = tail call double @cos(double noundef %272) #8, !tbaa !6
  %460 = tail call double @llvm.fmuladd.f64(double %266, double %459, double %447)
  %461 = fmul double %454, %460
  %462 = fmul double %444, 2.000000e+00
  %463 = fmul double %442, %462
  %464 = fneg double %451
  %465 = tail call double @llvm.fmuladd.f64(double %464, double %444, double %447)
  %466 = fmul double %429, %465
  %467 = tail call double @llvm.fmuladd.f64(double %451, double %442, double %446)
  %468 = fmul double %429, %467
  %469 = fmul double %455, %464
  %470 = fmul double %429, %469
  store double %466, ptr %2, align 8, !tbaa !10
  %471 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %472 = insertelement <2 x double> poison, double %470, i64 0
  %473 = shufflevector <2 x double> %472, <2 x double> poison, <2 x i32> zeroinitializer
  %474 = fmul <2 x double> %473, <double 0xBFD9752E50F4B399, double 0x3FED5C0357681EF3>
  %475 = insertelement <2 x double> poison, double %468, i64 0
  %476 = shufflevector <2 x double> %475, <2 x double> poison, <2 x i32> zeroinitializer
  %477 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %476, <2 x double> <double 0x3FED5C0357681EF3, double 0x3FD9752E50F4B399>, <2 x double> %474)
  store <2 x double> %477, ptr %471, align 8, !tbaa !10
  %478 = tail call double @llvm.fmuladd.f64(double %462, double %444, double -1.000000e+00)
  %479 = fmul double %463, %461
  %480 = tail call double @llvm.fmuladd.f64(double %478, double %458, double %479)
  %481 = fmul double %438, %480
  %482 = fmul double %442, -2.000000e+00
  %483 = tail call double @llvm.fmuladd.f64(double %482, double %442, double 1.000000e+00)
  %484 = fneg double %458
  %485 = fmul double %463, %484
  %486 = tail call double @llvm.fmuladd.f64(double %483, double %461, double %485)
  %487 = fmul double %438, %486
  %488 = fmul double %455, 2.000000e+00
  %489 = fmul double %442, %461
  %490 = tail call double @llvm.fmuladd.f64(double %444, double %458, double %489)
  %491 = fmul double %488, %490
  %492 = fmul double %438, %491
  %493 = getelementptr inbounds nuw i8, ptr %2, i64 24
  store double %481, ptr %493, align 8, !tbaa !10
  %494 = getelementptr inbounds nuw i8, ptr %2, i64 32
  %495 = insertelement <2 x double> poison, double %492, i64 0
  %496 = shufflevector <2 x double> %495, <2 x double> poison, <2 x i32> zeroinitializer
  %497 = fmul <2 x double> %496, <double 0xBFD9752E50F4B399, double 0x3FED5C0357681EF3>
  %498 = insertelement <2 x double> poison, double %487, i64 0
  %499 = shufflevector <2 x double> %498, <2 x double> poison, <2 x i32> zeroinitializer
  %500 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %499, <2 x double> <double 0x3FED5C0357681EF3, double 0x3FD9752E50F4B399>, <2 x double> %497)
  store <2 x double> %500, ptr %494, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #3

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare double @cos(double noundef) local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare double @sin(double noundef) local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare double @atan2(double noundef, double noundef) local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare double @sqrt(double noundef) local_unnamed_addr #2

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, errnomem: write) uwtable
define dso_local void @radecdist(ptr noundef readonly captures(none) %0, ptr noundef writeonly captures(none) initializes((0, 24)) %1) local_unnamed_addr #4 {
  %3 = load double, ptr %0, align 8, !tbaa !10
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %5 = load double, ptr %4, align 8, !tbaa !10
  %6 = fmul double %5, %5
  %7 = tail call double @llvm.fmuladd.f64(double %3, double %3, double %6)
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %9 = load double, ptr %8, align 8, !tbaa !10
  %10 = tail call double @llvm.fmuladd.f64(double %9, double %9, double %7)
  %11 = tail call double @llvm.sqrt.f64(double %10)
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store double %11, ptr %12, align 8, !tbaa !10
  %13 = load double, ptr %4, align 8, !tbaa !10
  %14 = load double, ptr %0, align 8, !tbaa !10
  %15 = tail call double @atan2(double noundef %13, double noundef %14) #8, !tbaa !6
  %16 = fmul double %15, 0x400E8EC8A4AEACC4
  %17 = fcmp olt double %16, 0.000000e+00
  %18 = fadd double %16, 2.400000e+01
  %19 = select i1 %17, double %18, double %16
  store double %19, ptr %1, align 8, !tbaa !10
  %20 = load double, ptr %8, align 8, !tbaa !10
  %21 = fdiv double %20, %11
  %22 = tail call double @asin(double noundef %21) #8, !tbaa !6
  %23 = fmul double %22, 0x404CA5DC1A63C1F8
  %24 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store double %23, ptr %24, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare double @asin(double noundef) local_unnamed_addr #2

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #5 {
  %3 = alloca [2 x double], align 8
  %4 = alloca [2 x [3 x double]], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #8
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #8
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %6 = getelementptr inbounds nuw i8, ptr %4, i64 8
  %7 = getelementptr inbounds nuw i8, ptr %4, i64 16
  br label %8

8:                                                ; preds = %2, %96
  %9 = phi i32 [ 0, %2 ], [ %97, %96 ]
  store double 0.000000e+00, ptr %5, align 8, !tbaa !10
  br label %10

10:                                               ; preds = %8, %10
  %11 = phi i32 [ 0, %8 ], [ %94, %10 ]
  %12 = phi double [ 0x4142B42C80000000, %8 ], [ %13, %10 ]
  %13 = fadd double %12, 1.000000e+00
  store double %13, ptr %3, align 8, !tbaa !10
  call void @planetpv(ptr noundef nonnull %3, i32 noundef 0, ptr noundef nonnull %4)
  %14 = load double, ptr %4, align 8, !tbaa !10
  %15 = load double, ptr %6, align 8, !tbaa !10
  %16 = fmul double %15, %15
  %17 = tail call double @llvm.fmuladd.f64(double %14, double %14, double %16)
  %18 = load double, ptr %7, align 8, !tbaa !10
  %19 = tail call double @llvm.fmuladd.f64(double %18, double %18, double %17)
  %20 = tail call double @llvm.sqrt.f64(double %19)
  %21 = tail call double @atan2(double noundef %15, double noundef %14) #8, !tbaa !6
  %22 = fdiv double %18, %20
  %23 = tail call double @asin(double noundef %22) #8, !tbaa !6
  call void @planetpv(ptr noundef nonnull %3, i32 noundef 1, ptr noundef nonnull %4)
  %24 = load double, ptr %4, align 8, !tbaa !10
  %25 = load double, ptr %6, align 8, !tbaa !10
  %26 = fmul double %25, %25
  %27 = tail call double @llvm.fmuladd.f64(double %24, double %24, double %26)
  %28 = load double, ptr %7, align 8, !tbaa !10
  %29 = tail call double @llvm.fmuladd.f64(double %28, double %28, double %27)
  %30 = tail call double @llvm.sqrt.f64(double %29)
  %31 = tail call double @atan2(double noundef %25, double noundef %24) #8, !tbaa !6
  %32 = fdiv double %28, %30
  %33 = tail call double @asin(double noundef %32) #8, !tbaa !6
  call void @planetpv(ptr noundef nonnull %3, i32 noundef 2, ptr noundef nonnull %4)
  %34 = load double, ptr %4, align 8, !tbaa !10
  %35 = load double, ptr %6, align 8, !tbaa !10
  %36 = fmul double %35, %35
  %37 = tail call double @llvm.fmuladd.f64(double %34, double %34, double %36)
  %38 = load double, ptr %7, align 8, !tbaa !10
  %39 = tail call double @llvm.fmuladd.f64(double %38, double %38, double %37)
  %40 = tail call double @llvm.sqrt.f64(double %39)
  %41 = tail call double @atan2(double noundef %35, double noundef %34) #8, !tbaa !6
  %42 = fdiv double %38, %40
  %43 = tail call double @asin(double noundef %42) #8, !tbaa !6
  call void @planetpv(ptr noundef nonnull %3, i32 noundef 3, ptr noundef nonnull %4)
  %44 = load double, ptr %4, align 8, !tbaa !10
  %45 = load double, ptr %6, align 8, !tbaa !10
  %46 = fmul double %45, %45
  %47 = tail call double @llvm.fmuladd.f64(double %44, double %44, double %46)
  %48 = load double, ptr %7, align 8, !tbaa !10
  %49 = tail call double @llvm.fmuladd.f64(double %48, double %48, double %47)
  %50 = tail call double @llvm.sqrt.f64(double %49)
  %51 = tail call double @atan2(double noundef %45, double noundef %44) #8, !tbaa !6
  %52 = fdiv double %48, %50
  %53 = tail call double @asin(double noundef %52) #8, !tbaa !6
  call void @planetpv(ptr noundef nonnull %3, i32 noundef 4, ptr noundef nonnull %4)
  %54 = load double, ptr %4, align 8, !tbaa !10
  %55 = load double, ptr %6, align 8, !tbaa !10
  %56 = fmul double %55, %55
  %57 = tail call double @llvm.fmuladd.f64(double %54, double %54, double %56)
  %58 = load double, ptr %7, align 8, !tbaa !10
  %59 = tail call double @llvm.fmuladd.f64(double %58, double %58, double %57)
  %60 = tail call double @llvm.sqrt.f64(double %59)
  %61 = tail call double @atan2(double noundef %55, double noundef %54) #8, !tbaa !6
  %62 = fdiv double %58, %60
  %63 = tail call double @asin(double noundef %62) #8, !tbaa !6
  call void @planetpv(ptr noundef nonnull %3, i32 noundef 5, ptr noundef nonnull %4)
  %64 = load double, ptr %4, align 8, !tbaa !10
  %65 = load double, ptr %6, align 8, !tbaa !10
  %66 = fmul double %65, %65
  %67 = tail call double @llvm.fmuladd.f64(double %64, double %64, double %66)
  %68 = load double, ptr %7, align 8, !tbaa !10
  %69 = tail call double @llvm.fmuladd.f64(double %68, double %68, double %67)
  %70 = tail call double @llvm.sqrt.f64(double %69)
  %71 = tail call double @atan2(double noundef %65, double noundef %64) #8, !tbaa !6
  %72 = fdiv double %68, %70
  %73 = tail call double @asin(double noundef %72) #8, !tbaa !6
  call void @planetpv(ptr noundef nonnull %3, i32 noundef 6, ptr noundef nonnull %4)
  %74 = load double, ptr %4, align 8, !tbaa !10
  %75 = load double, ptr %6, align 8, !tbaa !10
  %76 = fmul double %75, %75
  %77 = tail call double @llvm.fmuladd.f64(double %74, double %74, double %76)
  %78 = load double, ptr %7, align 8, !tbaa !10
  %79 = tail call double @llvm.fmuladd.f64(double %78, double %78, double %77)
  %80 = tail call double @llvm.sqrt.f64(double %79)
  %81 = tail call double @atan2(double noundef %75, double noundef %74) #8, !tbaa !6
  %82 = fdiv double %78, %80
  %83 = tail call double @asin(double noundef %82) #8, !tbaa !6
  call void @planetpv(ptr noundef nonnull %3, i32 noundef 7, ptr noundef nonnull %4)
  %84 = load double, ptr %4, align 8, !tbaa !10
  %85 = load double, ptr %6, align 8, !tbaa !10
  %86 = fmul double %85, %85
  %87 = tail call double @llvm.fmuladd.f64(double %84, double %84, double %86)
  %88 = load double, ptr %7, align 8, !tbaa !10
  %89 = tail call double @llvm.fmuladd.f64(double %88, double %88, double %87)
  %90 = tail call double @llvm.sqrt.f64(double %89)
  %91 = tail call double @atan2(double noundef %85, double noundef %84) #8, !tbaa !6
  %92 = fdiv double %88, %90
  %93 = tail call double @asin(double noundef %92) #8, !tbaa !6
  %94 = add nuw nsw i32 %11, 1
  %95 = icmp eq i32 %94, 36525
  br i1 %95, label %96, label %10, !llvm.loop !12

96:                                               ; preds = %10
  %97 = add nuw nsw i32 %9, 1
  %98 = icmp eq i32 %97, 20
  br i1 %98, label %99, label %8, !llvm.loop !14

99:                                               ; preds = %96
  %100 = fmul double %93, 0x404CA5DC1A63C1F8
  %101 = fmul double %91, 0x400E8EC8A4AEACC4
  %102 = fcmp olt double %101, 0.000000e+00
  %103 = fadd double %101, 2.400000e+01
  %104 = select i1 %102, double %103, double %101
  %105 = fmul double %83, 0x404CA5DC1A63C1F8
  %106 = fmul double %81, 0x400E8EC8A4AEACC4
  %107 = fcmp olt double %106, 0.000000e+00
  %108 = fadd double %106, 2.400000e+01
  %109 = select i1 %107, double %108, double %106
  %110 = fmul double %73, 0x404CA5DC1A63C1F8
  %111 = fmul double %71, 0x400E8EC8A4AEACC4
  %112 = fcmp olt double %111, 0.000000e+00
  %113 = fadd double %111, 2.400000e+01
  %114 = select i1 %112, double %113, double %111
  %115 = fmul double %63, 0x404CA5DC1A63C1F8
  %116 = fmul double %61, 0x400E8EC8A4AEACC4
  %117 = fcmp olt double %116, 0.000000e+00
  %118 = fadd double %116, 2.400000e+01
  %119 = select i1 %117, double %118, double %116
  %120 = fmul double %53, 0x404CA5DC1A63C1F8
  %121 = fmul double %51, 0x400E8EC8A4AEACC4
  %122 = fcmp olt double %121, 0.000000e+00
  %123 = fadd double %121, 2.400000e+01
  %124 = select i1 %122, double %123, double %121
  %125 = fmul double %43, 0x404CA5DC1A63C1F8
  %126 = fmul double %41, 0x400E8EC8A4AEACC4
  %127 = fcmp olt double %126, 0.000000e+00
  %128 = fadd double %126, 2.400000e+01
  %129 = select i1 %127, double %128, double %126
  %130 = fmul double %33, 0x404CA5DC1A63C1F8
  %131 = fmul double %31, 0x400E8EC8A4AEACC4
  %132 = fcmp olt double %131, 0.000000e+00
  %133 = fadd double %131, 2.400000e+01
  %134 = select i1 %132, double %133, double %131
  %135 = fmul double %23, 0x404CA5DC1A63C1F8
  %136 = fmul double %21, 0x400E8EC8A4AEACC4
  %137 = fcmp olt double %136, 0.000000e+00
  %138 = fadd double %136, 2.400000e+01
  %139 = select i1 %137, double %138, double %136
  %140 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %139, double noundef %135, double noundef %20)
  %141 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %134, double noundef %130, double noundef %30)
  %142 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %129, double noundef %125, double noundef %40)
  %143 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %124, double noundef %120, double noundef %50)
  %144 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %119, double noundef %115, double noundef %60)
  %145 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %114, double noundef %110, double noundef %70)
  %146 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %109, double noundef %105, double noundef %80)
  %147 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %104, double noundef %100, double noundef %90)
  %148 = load ptr, ptr @stdout, align 8, !tbaa !15
  %149 = tail call i32 @fflush(ptr noundef %148)
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #8
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #8
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #6

; Function Attrs: nofree nounwind
declare noundef i32 @fflush(ptr noundef captures(none)) local_unnamed_addr #6

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.sqrt.f64(double) #7

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x double> @llvm.fmuladd.v2f64(<2 x double>, <2 x double>, <2 x double>) #7

attributes #0 = { mustprogress nofree norecurse nounwind willreturn memory(errnomem: write) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(errnomem: write) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, errnomem: write) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"double", !8, i64 0}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.mustprogress"}
!14 = distinct !{!14, !13}
!15 = !{!16, !16, i64 0}
!16 = !{!"p1 _ZTS8_IO_FILE", !17, i64 0}
!17 = !{!"any pointer", !8, i64 0}
