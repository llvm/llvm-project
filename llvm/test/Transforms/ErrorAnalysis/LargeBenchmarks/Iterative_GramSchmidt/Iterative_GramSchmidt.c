#include <fenv.h>
#include <math.h>
#include <stdint.h>
#define TRUE 1
#define FALSE 0

float ex0(float Q11, float Q12, float Q13, float Q21, float Q22, float Q23, float Q31, float Q32, float Q33) {
	float eps = 5e-06f;
	float h1 = 0.0f;
	float h2 = 0.0f;
	float h3 = 0.0f;
	float qj1 = Q31;
	float qj2 = Q32;
	float qj3 = Q33;
	float r1 = 0.0f;
	float r2 = 0.0f;
	float r3 = 0.0f;
	float r = ((qj1 * qj1) + (qj2 * qj2)) + (qj3 * qj3);
	float rjj = 0.0f;
	float e = 10.0f;
	float i = 1.0f;
	float rold = sqrtf(r);
	int tmp = e > eps;
	while (tmp) {
		h1 = ((Q11 * qj1) + (Q21 * qj2)) + (Q31 * qj3);
		h2 = ((Q12 * qj1) + (Q22 * qj2)) + (Q32 * qj3);
		h3 = ((Q13 * qj1) + (Q23 * qj2)) + (Q33 * qj3);
		qj1 = qj1 - (((Q11 * h1) + (Q12 * h2)) + (Q13 * h3));
		qj2 = qj2 - (((Q21 * h1) + (Q22 * h2)) + (Q23 * h3));
		qj3 = qj3 - (((Q31 * h1) + (Q32 * h2)) + (Q33 * h3));
		r1 = r1 + h1;
		r2 = r2 + h2;
		r3 = r3 + h3;
		r = ((qj1 * qj1) + (qj2 * qj2)) + (qj3 * qj3);
		rjj = sqrtf(r);
		e = fabsf((1.0f - (rjj / rold)));
		i = i + 1.0f;
		rold = rjj;
		tmp = e > eps;
	}
	return qj1;
}

