#include <stdlib.h>

int main()
{
	int a[1000], b[1000];

	for (int i = 0; i < 1000; ++i)
		a[i] = i;
#pragma scop
	for (int i = 0; i < 1000; ++i)
		b[i] = a[i];
#pragma endscop
	for (int i = 0; i < 1000; ++i)
		if (b[i] != a[i])
			return EXIT_FAILURE;

	return EXIT_SUCCESS;
}
