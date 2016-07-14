#include <stdlib.h>

int main()
{
	int A[2][1000][1000];
	int B[2][1000][1000];

#pragma scop
	{
		for (int i = 0; i < 256; ++i)
			for (int j = 0; j < 256; ++j)
				if (j % 8 <= 2 || j % 8 >= 6)
					A[1][i][j] = B[1][j][i];
	}
#pragma endscop

/* 

When compiled with:

./ppcg tests/allow-sparse-copy-in.c --no-linearize-device-arrays
	--on-error=abort --sizes='{kernel[i]->tile[8,8]; kernel[i]->block[1,8]}'
	--max-shared-memory=-1  --unroll-copy-shared

this originally resulted in the following copy-in code:

      shared_B[0][0][t1] = B[1][8 * b1][8 * b0 + t1];
      shared_B[0][1][t1] = B[1][8 * b1 + 1][8 * b0 + t1];
      shared_B[0][2][t1] = B[1][8 * b1 + 2][8 * b0 + t1];
      shared_B[0][3][t1] = B[1][8 * b1 + 3][8 * b0 + t1];
      shared_B[0][4][t1] = B[1][8 * b1 + 4][8 * b0 + t1];
      shared_B[0][5][t1] = B[1][8 * b1 + 5][8 * b0 + t1];
      shared_B[0][6][t1] = B[1][8 * b1 + 6][8 * b0 + t1];
      shared_B[0][7][t1] = B[1][8 * b1 + 7][8 * b0 + t1];

whereas we only want to only perform copies that are actually needed:

      shared_B[0][0][t1] = B[1][8 * b1][8 * b0 + t1];
      shared_B[0][1][t1] = B[1][8 * b1 + 1][8 * b0 + t1];
      shared_B[0][2][t1] = B[1][8 * b1 + 2][8 * b0 + t1];
      shared_B[0][6][t1] = B[1][8 * b1 + 6][8 * b0 + t1];
      shared_B[0][7][t1] = B[1][8 * b1 + 7][8 * b0 + t1];
*/
	for (int i = 0; i < 100; ++i)
		if (A[1][0][i] != i)
			return EXIT_FAILURE;

	return EXIT_SUCCESS;
}
