; RUN: llc -verify-machineinstrs < %s -mattr=-vsx -mtriple=ppc32-- | grep fnabs

declare double @llvm.fabs.f64(double)

define double @test(double %X) {
        %Y = call double @llvm.fabs.f64( double %X ) readnone     ; <double> [#uses=1]
        %Z = fsub double -0.000000e+00, %Y               ; <double> [#uses=1]
        ret double %Z
}

