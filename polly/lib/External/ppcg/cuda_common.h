#ifndef _CUDA_COMMON_H_
#define _CUDA_COMMON_H_

#include <stdio.h>

struct cuda_info {
	FILE *host_c;
	FILE *kernel_c;
	FILE *kernel_h;
};

void cuda_open_files(struct cuda_info *info, const char *input);
void cuda_close_files(struct cuda_info *info);

#endif
