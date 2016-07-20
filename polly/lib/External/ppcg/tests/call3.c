#include <stdlib.h>

void copy_summary(int b[100], int a[100])
{
	for (int i = 0; i < 100; ++i) {
		b[i] = 0;
		int c = a[i];
	}
}

#ifdef pencil_access
__attribute__((pencil_access(copy_summary)))
#endif
void copy(int b[100], int a[100]);

int main()
{
	int A[100][100], B[100];

	for (int i = 0; i < 100; ++i)
		B[i] = i;
#pragma scop
	for (int i = 0; i < 100; ++i)
		copy(A[i], B);
#pragma endscop
	for (int i = 0; i < 100; ++i)
		for (int j = 0; j < 100; ++j)
			if (A[j][i] != B[i])
				return EXIT_FAILURE;

	return EXIT_SUCCESS;
}
