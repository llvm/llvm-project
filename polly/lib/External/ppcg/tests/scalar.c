#include <stdlib.h>

int main()
{
	int a;
#pragma scop
	a = 1;
#pragma endscop
	if (a != 1)
		return EXIT_FAILURE;

	return EXIT_SUCCESS;
}
