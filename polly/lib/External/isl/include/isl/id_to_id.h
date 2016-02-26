#ifndef ISL_ID_TO_ID_H
#define ISL_ID_TO_ID_H

#include <isl/id.h>

#define ISL_KEY_BASE	id
#define ISL_VAL_BASE	id
#include <isl/hmap.h>
#undef ISL_KEY_BASE
#undef ISL_VAL_BASE

#endif
