#include <stdlib.h>

struct s {
	int a;
	int b;
};

int main()
{
	int a[10];

	for (int i = 0; i < 10; ++i)
		a[i] = 0;
#pragma scop
	for (int i = 0; i < 10; ++i) {
		struct s b;
		b.a = 1;
		b.b = i;
		a[i] = b.a + b.b;
	}
#pragma endscop
	for (int i = 0; i < 10; ++i)
		if (a[i] != 1 + i)
			return EXIT_FAILURE;

	return EXIT_SUCCESS;
}
