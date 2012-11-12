; RUN: llc < %s -march=arm -mattr=+vfp2

define double @__ieee754_sqrt(double %x) {
	%tmp2 = tail call double asm "fsqrtd ${0:P}, ${1:P}", "=w,w"( double %x )
	ret double %tmp2
}

define float @__ieee754_sqrtf(float %x) {
	%tmp2 = tail call float asm "fsqrts $0, $1", "=w,w"( float %x )
	ret float %tmp2
}
