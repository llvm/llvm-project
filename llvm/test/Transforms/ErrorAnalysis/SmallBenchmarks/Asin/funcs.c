#include "funcs.h"
#include "table_generation.h"
#include <assert.h>
#include <math.h>

/*
 * Heavliy modified source from AMD's AOCL math library.
 * Modifications were made to allow the file to be compiled on its own.
 * In addition the kernel call to sqrt was replaced with a system level sqrt
 */

/*
 * Copyright (C) 2008-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

/*
 * ISO-IEC-10967-2: Elementary Numerical Functions
 * Signature:
 *   float asinf(float x)
 *

 * Contains implementation of float asinf(float x)
 *
 * The input domain should be in the [-1, +1] else a domain error is displayed
 *
 * asin(-x) = -asin(x)
 * asin(x) = pi/2-2*asin(sqrt(1/2*(1-x)))  when x > 1/2
 *
 * y = abs(x)
 * asinf(y) = asinf(g)  when y <= 0.5,  where g = y*y
 *		    = pi/2-asinf(g)  when y > 0.5, where g = 1/2*(1-y), y = -2*sqrt(g)
 * The term asin(f) is approximated by using a polynomial where the inputs lie in the interval [0 1/2]
 */

#include <stdint.h>
// #include <libm_util_amd.h>
#define EXPBITS_DP64      0x7ff0000000000000ULL
#define EXPSHIFTBITS_DP64 52
#define EXPBIAS_DP64      1023
// #include <libm/alm_special.h>
// #include <libm_macros.h>
// #include <libm/typehelper.h>
typedef    double              f64_t;
typedef union {
    f64_t    d;
    int64_t  i;
    uint64_t u;
} flt64_t;
static inline uint64_t
asuint64(double f)
{
	flt64_t fl = {.d = f};
	return fl.u;
}
// #include <libm/amd_funcs_internal.h>
// #include <libm/compiler.h>
// #include <libm/poly.h>
/*
 * p(x) = C1 + C2*r + C3*r^2 + C4*r^3 + C5*r^4 + C6*r^5 +
 *          C7*r^6 + C8*r^7 + C9*r^8 + C10*r^9 + C11*r^10 + C12*r^11
 *      = (C1 + C2*r) + r^2(C3 + C4*r) + r^4(C5 + C6*r) +
 *           r^6(C7 + C8*r) + r^8(C9 + C10*r) + r^10(C11 + C12*r)
 */
#define POLY_EVAL_12(x, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12) ({\
        __typeof(x) x2 =  x * x;                                             \
        __typeof(x) x4 = x2 * x2;                                            \
        __typeof(x) x8 = x4 * x4;                                            \
        __typeof(x) q, a0, a1, a2, a3, a4, a5, b0, b1, b2;                   \
         a0 =  c1 + c2  * x;                                                 \
         a1 =  c3 + c4  * x;                                                 \
         a2 =  c5 + c6  * x;                                                 \
         a3 =  c7 + c8  * x;                                                 \
         a4 =  c9 + c10 * x;                                                 \
         a5 = c11 + c12 * x;                                                 \
         b0 =  a0 + a1 * x2;                                                 \
         b1 =  a2 + a3 * x2;                                                 \
         b2 =  a4 + a5 * x2;                                                 \
                                                                             \
         q = (b0 + b1 * x4 ) + b2 * x8;                                      \
         q;                                                                  \
         })
// #include "kern/sqrt_pos.c"
// #include <libm/alm_special.h>
#include <math.h>

static struct {
    double half, piby2;
    double a[2], b[2], poly_asin[12];
} asin_data = {
    .half = 0x1p-1,
    .piby2      = 1.5707963267948965580e+00,
    // Values of factors of pi required to calculate asin
    .a = {
        0,
        0x1.921fb54442d18p0,
    },
    .b = {
        0x1.921fb54442d18p0,
        0x1.921fb54442d18p-1,
    },
    // Polynomial coefficients obtained using fpminimax algorithm from Sollya
    .poly_asin = {
        0x1.55555555552aap-3,
        0x1.333333337cbaep-4,
        0x1.6db6db3c0984p-5,
        0x1.f1c72dd86cbafp-6,
        0x1.6e89d3ff33aa4p-6,
        0x1.1c6d83ae664b6p-6,
        0x1.c6e1568b90518p-7,
        0x1.8f6a58977fe49p-7,
        0x1.a6ab10b3321bp-8,
        0x1.43305ebb2428fp-6,
        -0x1.0e874ec5e3157p-6,
        0x1.06eec35b3b142p-5,
    },
};

#define HALF asin_data.half

#define ALM_ASIN_PIBY2 asin_data.piby2
#define A asin_data.a
#define B asin_data.b

#define C1 asin_data.poly_asin[0]
#define C2 asin_data.poly_asin[1]
#define C3 asin_data.poly_asin[2]
#define C4 asin_data.poly_asin[3]
#define C5 asin_data.poly_asin[4]
#define C6 asin_data.poly_asin[5]
#define C7 asin_data.poly_asin[6]
#define C8 asin_data.poly_asin[7]
#define C9 asin_data.poly_asin[8]
#define C10 asin_data.poly_asin[9]
#define C11 asin_data.poly_asin[10]
#define C12 asin_data.poly_asin[11]



double
// ALM_PROTO_FAST(asin)(double x)
libm_dsl_amd_fast_asin(double x)
{
    double y, g, poly, result ,sign = 1.0 ;
    uint64_t ux;
    int64_t n = 0, xexp;

    // Include check for inf, -inf, nan here
    // asin(NaN) = NaN

    if (x < 0)
        sign = -1.0;

    ux   = asuint64 (x);

    /* Get the exponent part */
    xexp = (int)((ux & EXPBITS_DP64) >> EXPSHIFTBITS_DP64) - EXPBIAS_DP64;

    /* Special cases */
    if (x != x) {   /* if x is a nan */
        return x;
    }
    else if (xexp < -56) {
        /* Input small enough that arcsin(x) = x */
        return x;
    }
    else if (xexp >= 0) {
        /* abs(x) >= 1.0 */
        if (x == 1.0)
            return ALM_ASIN_PIBY2;
        else if (x == -1.0)
            return -ALM_ASIN_PIBY2;
        else
            return x;
    }


    y = sign*x;            // make x positive, if it is negative

    if (y > HALF)
    {
        n = 1 ;
        g = HALF*(1.0-y);
        //y = -2.0*ALM_PROTO_KERN(sqrt)(g);
        y = -2.0*sqrt(g) ;
    }
    else
    {
        g = y*y;
    }

    // Calculate the polynomial approximation x+C1*x^3+C2*x^5+C3*x^7+C4*x^9+C5*x^11+C6*x^13+C7*x^15+C8*x^17+C9*x^19+C10*x^21+C11*x^23+C12*x^25
    //                                       = x + x*(C1*x^2+C2*x^4+C3*x^6+C4*x^8+C5*x^10+C6*x^12+C7*x^14+C8*x^16+C9*x^18+C10*x^20+C11*x^22+C12*x^24)
    //                                       = x + x*(C1*G+C2*G^2+C3*G^3+C4*G^4+C5*G^5+C6*G^6+C7*G^7+C8*G^8+C9*G^9+C10*G^10+C11*G^11+C12*G^12)
    //                                       = x + x*G*(C1+G*(C2+G*(C3+G*(C4+G*(C5+G*(C6+G*(C7+G*(C8+G*(C9+G*(C10+G*(C11+C12*G)))))))))))

    poly = POLY_EVAL_12(g, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12);
    poly = y + y * g * poly;
    result =  poly+A[n] ;

    // if (sign == -1)  result = -result ;
    return (sign*result) ;
}

// strong_alias (__asin_finite, ALM_PROTO_FAST(asin))
// weak_alias (amd_asin, ALM_PROTO_FAST(asin))
// weak_alias (asin, ALM_PROTO_FAST(asin))

int mpfr_dsl_amd_fast_asin(mpfr_t out, double dx)
{
  static int init_called = 0;
  static mpfr_t x;
  if (!init_called) {
    mpfr_init2(x, ORACLE_PREC);
    init_called = 1;
  }
  mpfr_set_d(x, dx, MPFR_RNDN);
  mpfr_asin(out, x, MPFR_RNDN);
  return 0;
}
double dsl_amd_fast_asin(double x_in_1)
{
// (InflectionLeft (- x) (- y) (InflectionRight (sqrt (/ (- 1 x) 2)) (- (/ PI 2) (* 2 y)) (Horner (MinimaxPolynomial (asin x) [0 0.5]))))
    double x_in_0 = x_in_1 < 0.0 ? (-x_in_1) : x_in_1;
    double horner_in_0 = x_in_0 < 0.5 ? x_in_0 : sqrt(((1-x_in_0)/2));
    double horner_out_0 = (((double)horner_in_0)*(((double)0x1.0000000000000ef89db007982692a018p0)
        + ((double)horner_in_0)*((double)horner_in_0)*(((double)0x1.55555555547656621439c1510aede0b6p-3)
        + ((double)horner_in_0)*((double)horner_in_0)*(((double)0x1.333333342971b9b22f533fdf1cf7d8e6p-4)
        + ((double)horner_in_0)*((double)horner_in_0)*(((double)0x1.6db6daef4023cb111c715383da8ff0cap-5)
        + ((double)horner_in_0)*((double)horner_in_0)*(((double)0x1.f1c740f8e800dd53f6b482edc0ce51ap-6)
        + ((double)horner_in_0)*((double)horner_in_0)*(((double)0x1.6e885ada8fedbd0e1a1143843637f44p-6)
        + ((double)horner_in_0)*((double)horner_in_0)*(((double)0x1.1c807f773bbe6036ba4de0656e0b86ecp-6)
        + ((double)horner_in_0)*((double)horner_in_0)*(((double)0x1.c5986c3e05586fd823bce474205cc084p-7)
        + ((double)horner_in_0)*((double)horner_in_0)*(((double)0x1.96f49a09ac2b2d0b0c2fbce7420b9a7cp-7)
        + ((double)horner_in_0)*((double)horner_in_0)*(((double)0x1.6b884638d2bf7d1e12ec60d48db28e8p-8)
        + ((double)horner_in_0)*((double)horner_in_0)*(((double)0x1.6849abd2494cf9f3f02498e0340701c2p-6)
        + ((double)horner_in_0)*((double)horner_in_0)*(((double)-0x1.4464f1fe9547639c3886d71db3bce2f6p-6)
        + ((double)horner_in_0)*((double)horner_in_0)*((double)0x1.18243752d95434036b3afbde665c0a58p-5))))))))))))));
    double y_out_0 = x_in_0 < 0.5 ? horner_out_0 : ((M_PI/2)-(2*horner_out_0));
    double y_out_1 = x_in_1 < 0.0 ? (-y_out_0) : y_out_0;
    return y_out_1;
}

double hand_worked(double x_in_1)
{
// (InflectionLeft (- x) (- y) (InflectionRight (sqrt (/ (- 1 x) 2)) (- (/ PI 2) (* 2 y)) (Horner (MinimaxPolynomial (asin x) [0 0.5]))))
    double x_in_0 = x_in_1 < 0.0 ? (-x_in_1) : x_in_1;
    double horner_in_0 = x_in_0 < 0.5 ? x_in_0 : sqrt(((1-x_in_0)/2));
    double horner_in_0_2 = horner_in_0 * horner_in_0;
    double horner_out_0 = (((double)horner_in_0)*(((double)1)
        + horner_in_0_2*(((double)0.1666666666666477004)
        + horner_in_0_2*(((double)0.07500000000417969548)
        + horner_in_0_2*(((double)0.04464285678140855751)
        + horner_in_0_2*(((double)0.03038196065035564039)
        + horner_in_0_2*(((double)0.0223717279703189581)
        + horner_in_0_2*(((double)0.01736009463784134871)
        + horner_in_0_2*(((double)0.01388184285963460496)
        + horner_in_0_2*(((double)0.01218919111033679899)
        + horner_in_0_2*(((double)0.00644940526689945226)
        + horner_in_0_2*(((double)0.01972588778568478904)
        + horner_in_0_2*(((double)-0.01651175205874840998)
        + horner_in_0_2*((double)0.03209627299824770186))))))))))))));
    double y_out_0 = x_in_0 < 0.5 ? horner_out_0 : ((M_PI/2)-(2*horner_out_0));
    double y_out_1 = x_in_1 < 0.0 ? (-y_out_0) : y_out_0;
    return y_out_1;
}
