#include <stdlib.h>

/* Check that the sources of live ranges with the same sink
 * are executed in order.
 */
int main()
{
	int A[128];
	int n = 128;

	A[0] = 0;
#pragma scop
	for (int i = 0; i < n; ++i) {
		int set = 0;
		if (A[i] < 2)
			set = 1;
		if (set)
			A[i] = 2;
	}
#pragma endscop
	if (A[0] != 2)
		return EXIT_FAILURE;

	return EXIT_SUCCESS;
}
