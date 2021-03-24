; RUN: %lli %s > /dev/null

define i32 @main() {
    %double1 = fadd <2 x double> <double 0.0, double 0.0>, <double 0.0, double 0.0>
    %double2 = fadd <2 x double> <double 0.0, double 0.0>, <double 0.0, double 0.0>
    %float1 = fadd <3 x float> <float 0.0, float 0.0, float 0.0>, <float 0.0, float 0.0, float 0.0>
    %float2 = fadd <3 x float> <float 0.0, float 0.0, float 0.0>, <float 0.0, float 0.0, float 0.0>
    %test49 = fcmp oeq <3 x float> %float1, %float2
    %test50 = fcmp oge <3 x float> %float1, %float2
    %test51 = fcmp ogt <3 x float> %float1, %float2
    %test52 = fcmp ole <3 x float> %float1, %float2
    %test53 = fcmp olt <3 x float> %float1, %float2
    %test54 = fcmp une <3 x float> %float1, %float2

    %test55 = fcmp oeq <2 x double> %double1, %double2
    %test56 = fcmp oge <2 x double> %double1, %double2
    %test57 = fcmp ogt <2 x double> %double1, %double2
    %test58 = fcmp ole <2 x double> %double1, %double2
    %test59 = fcmp olt <2 x double> %double1, %double2
    %test60 = fcmp une <2 x double> %double1, %double2

    ret i32 0
}


