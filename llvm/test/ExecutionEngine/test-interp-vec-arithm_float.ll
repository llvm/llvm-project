; RUN: %lli %s > /dev/null


define i32 @main() {

    %A_float = fadd <4 x float> <float 0.0, float 11.0, float 22.0, float 33.0>, <float 44.0, float 55.0, float 66.0, float 77.0>
    %B_float = fsub <4 x float> %A_float, <float 88.0, float 99.0, float 100.0, float 111.0>
    %C_float = fmul <4 x float> %B_float, %B_float
    %D_float = fdiv <4 x float> %C_float, %B_float
    %E_float = frem <4 x float> %D_float, %A_float
    %F_float = fneg <4 x float> %E_float


    %A_double = fadd <3 x double> <double 0.0, double 111.0, double 222.0>, <double 444.0, double 555.0, double 665.0>
    %B_double = fsub <3 x double> %A_double, <double 888.0, double 999.0, double 1001.0>
    %C_double = fmul <3 x double> %B_double, %B_double
    %D_double = fdiv <3 x double> %C_double, %B_double
    %E_double = frem <3 x double> %D_double, %A_double
    %F_double = fneg <3 x double> %E_double

    ret i32 0
}
