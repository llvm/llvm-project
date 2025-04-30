#ifndef NEXT_MEMORY_STRUCTURE_H
#define NEXT_MEMORY_STRUCTURE_H

#include <stdint.h>

/* csupport definitions
 * We prefix these macros as a namespacing mechanism to prevent conflicts
 * with other headers in nextutils
 */
#define _NM_BIT(nr) (1UL << (nr))
#define _NM_KIB(x) ((x)*1024)
#define _NM_MIB(x) (_NM_KIB(x) * 1024)
#define _NM_GIB(x) (_NM_MIB(x) * 1024L)
#define _NM_TIB(x) (_NM_GIB(x) * 1024L)

#ifdef __cplusplus
#define REINTERPRET_CAST(type, x) reinterpret_cast<type>(x)
#define STATIC_CAST(type, x) static_cast<type>(x)
#else
#define REINTERPRET_CAST(type, x) ((type)(x))
#define STATIC_CAST(type, x) ((type)(x))
#endif

/* Next memory regions configuration
 *
 * Note that we have two constraints about these addresses:
 * - RISC supports addresses that consist of maximum 44 bits, with the last bit
 *   being sign-extended. Since x86 Linux mandates MSB in user mode addresses
 *   to be off, this means that the 44th bit must also be off, and limits us to
 *   43 bits addresses.
 * - When compiling with ASAN enabled, it has hardcoded addresses it uses for
 *   x86_64.
 *   (reference: https://github.com/google/sanitizers/wiki/AddressSanitizerAlgorithm#64-bit)
 *   These addresses end up taking most of the range accessible through 43
 *   bits. The only part we can somehow use out of this is the ShadowGap, whose
 *   range is [0x004091ff7000, 0x02008fff6fff].
 *   Ideally we'd want to find a way to overcome the ASAN constraint, but for
 *   now we simply put our addresses in this range.
 */

typedef struct {
    uintptr_t start;
    uint64_t size;
} next_memory_section;

/* clang-format off */
#define __NEXT_MEMORY_ALLOCATOR            (next_memory_section){.start = 0x0000008000000000, .size = 0x0000008000000000 /*  512 GiB */}

#define NEXT_MEMORY_SECTION_START_PTR(section) (REINTERPRET_CAST(void *, (section).start))
#define NEXT_MEMORY_SECTION_END_PTR(section)   (REINTERPRET_CAST(void *, ((section).start + (section).size)))

#define NEXT_MEMORY_ALLOCATOR_BASE     (NEXT_MEMORY_SECTION_START_PTR(__NEXT_MEMORY_ALLOCATOR))
#define NEXT_MEMORY_ALLOCATOR_END      (NEXT_MEMORY_SECTION_END_PTR(__NEXT_MEMORY_ALLOCATOR))

/* clang-format on */

/* Next memory regions derived values */
#define NEXT_MEMORY_CLASS(size_class_bits)                                                         \
    REINTERPRET_CAST(void *,                                                                       \
                     REINTERPRET_CAST(uintptr_t, NEXT_MEMORY_ALLOCATOR_BASE) |                     \
                         ((STATIC_CAST(uintptr_t, size_class_bits) - SIZE_CLASS_MINIMAL_BITS)      \
                          << SIZE_CLASS_SIZE_BITS))
#define NEXT_MEMORY_CLASS_WILD                                                                     \
    REINTERPRET_CAST(void *, REINTERPRET_CAST(uintptr_t, NEXT_MEMORY_ALLOCATOR_BASE) |             \
                                 (SIZE_CLASS_INDEX_WILD << SIZE_CLASS_SIZE_BITS))

/* Next allocator configuration */
#define SIZE_CLASS_MINIMAL_BITS 12
#define SIZE_CLASS_MAXIMAL_BITS 17
#define SIZE_CLASS_SIZE_BITS 36
#define SIZE_CLASS_INDEX_BITS 3
#define POOL_ALLOCATION_COUNT_BITS 10

/* Next allocator derived values */
#define SIZE_CLASS_MINIMAL _NM_BIT(SIZE_CLASS_MINIMAL_BITS)
#define SIZE_CLASS_MINIMAL_MASK (SIZE_CLASS_MINIMAL - 1)
#define SIZE_CLASS_MINIMAL_MASK_INV ~SIZE_CLASS_MINIMAL_MASK
#define SIZE_CLASS_MAXIMAL _NM_BIT(SIZE_CLASS_MAXIMAL_BITS)
#define SIZE_CLASS_COUNT (SIZE_CLASS_MAXIMAL_BITS - SIZE_CLASS_MINIMAL_BITS + 1)
#define SIZE_CLASS_SIZE _NM_BIT(SIZE_CLASS_SIZE_BITS)
#define SIZE_CLASS_MASK (SIZE_CLASS_SIZE - 1)
#define SIZE_CLASS_MASK_INV ~SIZE_CLASS_MASK
#define SIZE_CLASS_INDEX_MASK (_NM_BIT(SIZE_CLASS_INDEX_BITS) - 1)
#define SIZE_CLASS_INDEX_WILD SIZE_CLASS_INDEX_MASK
#define POOL_ALLOCATION_COUNT _NM_BIT(POOL_ALLOCATION_COUNT_BITS)
#define POOL_ALLOCATION_MASK (POOL_ALLOCATION_COUNT - 1)
#define POOL_ALLOCATION_MASK_INV ~POOL_ALLOCATION_MASK
#define POOLS_COUNT_BITS(size_class_bits)                                                          \
    (SIZE_CLASS_SIZE_BITS - POOL_ALLOCATION_COUNT_BITS - (size_class_bits))
#define POOLS_COUNT(size_class_bits) _NM_BIT(POOLS_COUNT_BITS(size_class_bits))

/* Next allocator pointer size class calculations */
#define POINTER_SIZE_CLASS_INDEX(pointer)                                                          \
    ((REINTERPRET_CAST(uintptr_t, pointer) >> SIZE_CLASS_SIZE_BITS) & SIZE_CLASS_INDEX_MASK)
#define POINTER_SIZE_CLASS_BITS(pointer)                                                           \
    (POINTER_SIZE_CLASS_INDEX(pointer) + SIZE_CLASS_MINIMAL_BITS)
#define POINTER_SIZE_CLASS(pointer) _NM_BIT(POINTER_SIZE_CLASS_BITS(pointer))

/* Next allocator memory structures */
struct size_class_data {
    uint64_t memory_entry;
    uint32_t pool_free_counters[POOLS_COUNT(SIZE_CLASS_MINIMAL_BITS)];
};

struct next_memory_context {
    struct size_class_data size_classes[SIZE_CLASS_COUNT];
    uint32_t big_heap_lock;
};
#endif /* NEXT_MEMORY_STRUCTURE_H */
