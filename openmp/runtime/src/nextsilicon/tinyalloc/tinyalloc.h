#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>

/* NOLINTNEXTLINE(readability-avoid-const-params-in-decls) */
bool ta_init(void *base, const void *limit, const size_t heap_blocks,
             const size_t split_thresh, const size_t alignment);
void *ta_alloc(size_t num);
void *ta_calloc(size_t num, size_t size);
bool ta_free(void *ptr);

size_t ta_num_free();
size_t ta_num_used();
size_t ta_num_fresh();
bool ta_check();

#ifdef __cplusplus
}
#endif
