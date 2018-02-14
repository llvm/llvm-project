#include <stdlib.h>

/* Check that a write access is not removed from the live-out
 * accesses only because a strict subset of the (potentially)
 * accessed elements are killed by a later write.
 */
int main()
{
	int A[10];

	A[1] = 0;
#pragma scop
	int i = 1;
	i = i * i;
	A[i] = 1;
	A[0] = 0;
#pragma endscop
	if (A[1] != 1)
		return EXIT_FAILURE;

	return EXIT_SUCCESS;
}
