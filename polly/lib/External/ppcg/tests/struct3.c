#include <stdlib.h>

struct s {
	int a;
	int b;
};

int main()
{
	struct s a, b[10];

	a.b = 57;
#pragma scop
	a.a = 42;
	for (int i = 0; i < 10; ++i)
		b[i] = a;
#pragma endscop
	for (int i = 0; i < 10; ++i)
		if (b[i].a != 42)
			return EXIT_FAILURE;
	if (a.b != 57)
		return EXIT_FAILURE;

	return EXIT_SUCCESS;
}
