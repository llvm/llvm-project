#include "gitversion.h"

const char *ppcg_version(void)
{
	return GIT_HEAD_ID"\n";
}
