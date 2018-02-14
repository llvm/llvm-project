#include <stdlib.h>

int main()
{
	int A[100];

#pragma scop
	{
		int B[100];
		B[0] = 0;
		for (int i = 1; i < 100; ++i)
			B[i] = B[i - 1] + 1;
		for (int i = 0; i < 100; ++i)
			A[i] = B[i];
	}
#pragma endscop
	for (int i = 0; i < 100; ++i)
		if (A[i] != i)
			return EXIT_FAILURE;

	return EXIT_SUCCESS;
}
