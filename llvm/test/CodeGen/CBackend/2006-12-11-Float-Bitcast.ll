; RUN: llvm-as < %s | llc -march=c | \
; RUN:   grep __BITCAST | count 14

define i32 @test1(float %F) {
        %X = bitcast float %F to i32            ; <i32> [#uses=1]
        ret i32 %X
}

define float @test2(i32 %I) {
        %X = bitcast i32 %I to float            ; <float> [#uses=1]
        ret float %X
}

define i64 @test3(double %D) {
        %X = bitcast double %D to i64           ; <i64> [#uses=1]
        ret i64 %X
}

define double @test4(i64 %L) {
        %X = bitcast i64 %L to double           ; <double> [#uses=1]
        ret double %X
}

define double @test5(double %D) {
        %X = bitcast double %D to double                ; <double> [#uses=1]
        %Y = add double %X, 2.000000e+00                ; <double> [#uses=1]
        %Z = bitcast double %Y to i64           ; <i64> [#uses=1]
        %res = bitcast i64 %Z to double         ; <double> [#uses=1]
        ret double %res
}

define float @test6(float %F) {
        %X = bitcast float %F to float          ; <float> [#uses=1]
        %Y = add float %X, 2.000000e+00         ; <float> [#uses=1]
        %Z = bitcast float %Y to i32            ; <i32> [#uses=1]
        %res = bitcast i32 %Z to float          ; <float> [#uses=1]
        ret float %res
}

define i32 @main(i32 %argc, i8** %argv) {
        %a = call i32 @test1( float 0x400921FB40000000 )                ; <i32> [#uses=2]
        %b = call float @test2( i32 %a )                ; <float> [#uses=0]
        %c = call i64 @test3( double 0x400921FB4D12D84A )               ; <i64> [#uses=1]
        %d = call double @test4( i64 %c )               ; <double> [#uses=0]
        %e = call double @test5( double 7.000000e+00 )          ; <double> [#uses=0]
        %f = call float @test6( float 7.000000e+00 )            ; <float> [#uses=0]
        ret i32 %a
}

