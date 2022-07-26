#include<stdio.h>
#include<math.h>

int main() {
    float a, b;
    double c, d;

    float f_sin, f_cos, f_tan, f_asin, f_acos, f_atan, f_sinh, f_cosh, f_tanh,
        f_exp, f_log, f_sqrt, f_truncToFloat, f_add, f_sub, f_mul, f_div;
    float d_sin, d_cos, d_tan, d_asin, d_acos, d_atan, d_sinh, d_cosh, d_tanh,
        d_exp, d_log, d_sqrt, d_add, d_sub, d_mul, d_div;
    
    printf("Enter value of a: ");
    scanf("%f", &a);
    printf("Enter value of b: ");
    scanf("%f", &b);
    printf("Enter value of c: ");
    scanf("%lf", &c);
    printf("Enter value of d: ");
    scanf("%lf", &d);

    // %.2lf displays number up to 2 decimal point
    // FP32
    // Unary Operations
    f_sin = sinf(a);
    printf("Sin = %.2f\n", f_sin);
    fAFfp32markForResult(f_sin);

    f_cos = cosf(a);
    printf("Cos = %.2f\n", f_cos);
    fAFfp32markForResult(f_cos);

    f_tan = tanf(a);
    printf("Tan = %.2f\n", f_tan);
    fAFfp32markForResult(f_tan);

    f_asin = asinf(a);
    printf("ArcSin = %.2f\n", f_asin);
    fAFfp32markForResult(f_asin);

    f_acos = acosf(a);
    printf("ArcCos = %.2f\n", f_acos);
    fAFfp32markForResult(f_acos);

    f_atan = atanf(a);
    printf("ArcTan = %.2f\n", f_atan);
    fAFfp32markForResult(f_atan);

    f_sinh = sinhf(a);
    printf("Sinh = %.2f\n", sinhf(a));
    fAFfp32markForResult(f_sinh);

    f_cosh = coshf(a);
    printf("Cosh = %.2f\n", f_cosh);
    fAFfp32markForResult(f_cosh);

    f_tanh = tanhf(a);
    printf("Tanh = %.2f\n", f_tanh);
    fAFfp32markForResult(f_tanh);

    f_exp = expf(a);
    printf("Exp = %.2f\n", expf(a));
    fAFfp32markForResult(f_exp);

    f_log = logf(a);
    printf("Log = %.2f\n", f_log);
    fAFfp32markForResult(f_log);

    f_sqrt = sqrtf(a);
    printf("Sqrt = %.2f\n", f_sqrt);
    fAFfp32markForResult(f_sqrt);

    // Binary Operations
    f_add = a+b;
    printf("Sum = %.2f\n", f_add);
    fAFfp32markForResult(f_add);

    f_sub = a-b;
    printf("Difference = %.2f\n", f_sub);
    fAFfp32markForResult(f_sub);

    f_mul = a*b;
    printf("Product = %.2f\n", f_mul);
    fAFfp32markForResult(f_mul);

    f_div = a/b;
    printf("Quotient = %.2f\n", f_div);
    fAFfp32markForResult(f_div);

    // FP64
    // Unary Operations
    d_sin = sin(c);
    printf("Sin = %2lf\n", d_sin);
    fAFfp64markForResult(d_sin);

    d_cos = cos(c);
    printf("Cos = %2lf\n", d_cos);
    fAFfp64markForResult(d_cos);

    d_tan = tan(c);
    printf("Tan = %2lf\n", d_tan);
    fAFfp64markForResult(d_tan);

    d_asin = asin(c);
    printf("ArcSin = %2lf\n", d_asin);
    fAFfp64markForResult(d_asin);

    d_acos = acos(c);
    printf("ArcCos = %2lf\n", d_acos);
    fAFfp64markForResult(d_acos);

    d_atan = atan(c);
    printf("ArcTan = %2lf\n", d_atan);
    fAFfp64markForResult(d_atan);

    d_sinh = sinh(c);
    printf("Sinh = %2lf\n", d_sinh);
    fAFfp64markForResult(d_sinh);

    d_cosh = cosh(c);
    printf("Cosh = %2lf\n", d_cosh);
    fAFfp64markForResult(d_cosh);

    d_tanh = tanh(c);
    printf("Tanh = %2lf\n", d_tanh);
    fAFfp64markForResult(d_tanh);

    d_exp = exp(c);
    printf("Exp = %2lf\n", d_exp);
    fAFfp64markForResult(d_exp);

    d_log = log(c);
    printf("Log = %2lf\n", d_log);
    fAFfp64markForResult(d_log);

    d_sqrt = sqrt(c);
    printf("Sqrt = %2lf\n", d_sqrt);
    fAFfp64markForResult(d_sqrt);

    f_truncToFloat = (float)c;
    printf("TruncToFloat = %f\n", f_truncToFloat);

    // Binary Operations
    d_add = c+d;
    printf("Sum = %.2lf\n", d_add);
    fAFfp64markForResult(d_add);

    d_sub = c-d;
    printf("Difference = %2lf\n", d_sub);
    fAFfp64markForResult(d_sub);

    d_mul = c*d;
    printf("Product = %.2lf\n", d_mul);
    fAFfp64markForResult(d_mul);

    d_div = c/d;
    printf("Quotient = %2lf\n", d_div);
    fAFfp64markForResult(d_div);

    return 0;
}

