#include <stdlib.h>

struct s {
	int c[10][10];
};

int main()
{
	struct s a[10][10], b[10][10];

	for (int i = 0; i < 10; ++i)
		for (int j = 0; j < 10; ++j)
			for (int k = 0; k < 10; ++k)
				for (int l = 0; l < 10; ++l)
					a[i][j].c[k][l] = i + j + k + l;
#pragma scop
	for (int i = 0; i < 10; ++i)
		for (int j = 0; j < 10; ++j)
			for (int k = 0; k < 10; ++k)
				for (int l = 0; l < 10; ++l)
					b[i][j].c[k][l] = i + j + k + l;
#pragma endscop
	for (int i = 0; i < 10; ++i)
		for (int j = 0; j < 10; ++j)
			for (int k = 0; k < 10; ++k)
				for (int l = 0; l < 10; ++l)
					if (b[i][j].c[k][l] != a[i][j].c[k][l])
						return EXIT_FAILURE;

	return EXIT_SUCCESS;
}
