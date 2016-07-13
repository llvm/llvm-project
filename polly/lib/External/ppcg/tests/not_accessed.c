#include <stdlib.h>

void copy_summary(int b[1000], int a[1000], int pos, int c[1000])
{
	b[pos] = 0;
	int d = a[pos];
}

#ifdef pencil_access
__attribute__((pencil_access(copy_summary)))
#endif
void copy(int b[1000], int a[1000], int pos, int c[1000]);

int main()
{
	int a[1000], b[1000], c[1000];

	for (int i = 0; i < 1000; ++i)
		a[i] = i;
#pragma scop
	for (int i = 0; i < 1000; ++i)
		copy(b, a, i, c);
#pragma endscop
	for (int i = 0; i < 1000; ++i)
		if (b[i] != a[i])
			return EXIT_FAILURE;

	return EXIT_SUCCESS;
}
