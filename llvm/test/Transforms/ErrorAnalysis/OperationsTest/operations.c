#include<stdio.h>
#include<math.h>

int main() {
    float a, b;
    double c, d;
    
    printf("Enter value of a: ");
    scanf("%f", &a);
    printf("Enter value of b: ");
    scanf("%f", &b);
    printf("Enter value of c: ");
    scanf("%lf", &c);
    printf("Enter value of c: ");
    scanf("%lf", &d);

    // %.2lf displays number up to 2 decimal point
    // FP32
    // Unary Operations
    printf("Sin = %2f\n", sin(a));
    printf("Cos = %2f\n", cos(a));
    printf("Tan = %2f\n", tan(a));
    printf("ArcSin = %2f\n", asin(a));
    printf("ArcCos = %2f\n", acos(a));
    printf("ArcTan = %2f\n", atan(a));
    printf("Sinh = %2f\n", sinh(a));
    printf("Cosh = %2f\n", cosh(a));
    printf("Tanh = %2f\n", tanh(a));
    printf("Exp = %2f\n", exp(a));
    printf("Log = %2f\n", log(a));
    printf("Sqrt = %2f\n", sqrt(a));

    // Binary Operations
    printf("Sum = %.2f\n", a+b);
    printf("Difference = %2f\n", a-b);
    printf("Product = %.2f\n", a*b);
    printf("Quotient = %2f\n", a/b);

    // FP64
    // Unary Operations
    printf("Sin = %2lf\n", sin(c));
    printf("Cos = %2lf\n", cos(c));
    printf("Tan = %2lf\n", tan(c));
    printf("ArcSin = %2lf\n", asin(c));
    printf("ArcCos = %2lf\n", acos(c));
    printf("ArcTan = %2lf\n", atan(c));
    printf("Sinh = %2lf\n", sinh(c));
    printf("Cosh = %2lf\n", cosh(c));
    printf("Tanh = %2lf\n", tanh(c));
    printf("Exp = %2lf\n", exp(c));
    printf("Log = %2lf\n", log(c));
    printf("Sqrt = %2lf\n", sqrt(c));
    printf("TruncToFloat = %f\n", (float)c);

    // Binary Operations
    printf("Sum = %.2lf\n", c+d);
    printf("Difference = %2lf\n", c-d);
    printf("Product = %.2lf\n", c*d);
    printf("Quotient = %2lf\n", c/d);
    
    return 0;
}

