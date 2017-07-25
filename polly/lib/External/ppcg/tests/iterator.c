#include <stdlib.h>

int main()
{
	int i;
	int a[101];

	i = 0;
#pragma scop
	for (i = 0; i < 100; ++i)
		a[i] = i;
	a[i] = i;
#pragma endscop
	if (a[100] != 100)
		return EXIT_FAILURE;

	return EXIT_SUCCESS;
}
