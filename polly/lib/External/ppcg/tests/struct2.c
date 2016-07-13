#include <stdlib.h>

struct s {
	int a;
};

int main()
{
	struct s a, b[10];

#pragma scop
	a.a = 42;
	for (int i = 0; i < 10; ++i)
		b[i].a = a.a;
#pragma endscop
	for (int i = 0; i < 10; ++i)
		if (b[i].a != 42)
			return EXIT_FAILURE;

	return EXIT_SUCCESS;
}
