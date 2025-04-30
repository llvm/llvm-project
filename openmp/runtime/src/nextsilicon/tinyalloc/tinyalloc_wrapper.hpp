#ifndef __TINYALLOC_WRAPPER_HPP__
#define __TINYALLOC_WRAPPER_HPP__

/// @file tinyalloc_wrapper.hpp
///
/// @ref https://github.com/thi-ng/tinyalloc
///
/// Tinyalloc is a simple heap allocator, which manages a single
/// block of memory.
/// Originally, it comes as a single instance with no guard.
///
/// The wrapper is still single-instance (no extensions).
/// It is mutex guarded, and heap size can be set with env var:
///
/// name                         default
/// __TINYALLOC_HEAP_SIZE        64MiB
/// __TINYALLOC_HEAP_MAX_CHUNKS  64ki
#include <cstddef>

namespace ns {
namespace tinyalloc {
namespace wrapper {

void ensure_initialized();
void *alloc(size_t size);
void *calloc(size_t num, size_t size);
void *realloc(void *ptr, size_t size);
void free(void *addr);

} // namespace wrapper
} // namespace tinyalloc
} // namespace ns

#endif // __TINYALLOC_WRAPPER_HPP__
