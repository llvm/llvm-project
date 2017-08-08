#ifndef UTIL_H
#define UTIL_H

#include <string.h>

/* Compare the prefix of "s" to "prefix" up to the length of "prefix".
 */
static inline int prefixcmp(const char *s, const char *prefix)
{
	return strncmp(s, prefix, strlen(prefix));
}

#endif
