#include "kmp.h"
#include "kmp_debug.h"
#include "kmp_ns_mark_page.h"

#include "tinyalloc_wrapper.hpp"
#include "tinyalloc.h"

#include <sys/mman.h>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <iostream>

#if LIBOMP_NEXTSILICON_ATOMICS_BYPASS

const static char *HEAP_SIZE_ENV = "__TINYALLOC_HEAP_SIZE";
const static char *HEAP_NUM_CHUNKS_ENV = "__TINYALLOC_HEAP_MAX_CHUNKS";

constexpr static size_t HEAP_DEFAULT_SIZE = 512ULL * 1024ULL * 1024ULL;
constexpr static size_t HEAP_MAX_NUM_CHUNKS = 512UL * 1024UL;
constexpr static size_t HEAP_SPLIT_THRESHOLD = 16;
constexpr static size_t HEAP_ALIGNMENT = 16;

void __kmp_ns_mark_page(void *addr, size_t size, const char *caller);

/// @brief tinyalloc is a simple single arena heap allocator.
/// the wrapper provides:
/// 1. mutex guard
/// 2. env confiruation (defaulted to 64MiB)
///
/// tinyalloc heap provides simple heap alloc/free service
/// in places where the libc's malloc/free are forbidden.
///
/// @note There's no explicit destruction path for tinyalloc heap.
/// I.e, once created, it only goes down with the process. This
/// simplifies handling termination sequences and there's little point
/// in cleaning this up just when the process is about to end anyway.
namespace ns {
namespace tinyalloc {
namespace wrapper {

struct tinyalloc_threadsafe {
private:
  uint8_t *_heap;
  size_t _heap_size;
  std::mutex _mutex;

  void *alloc_impl(size_t size) {
    // prepare for realloc - keep header for size.
    auto new_size = size + sizeof(size_t);

    auto *res = ta_alloc(new_size);
    if (!res) {
      std::cerr << "ta_alloc() failed for [" << new_size
                << "] bytes - no memory\n";
      return nullptr;
    }

    size_t *first_word = reinterpret_cast<size_t *>(res);
    first_word[0] = size;
    return &first_word[1];
  }

  void *calloc_impl(size_t num, size_t size) {
    // prepare for realloc - keep header for size.
    auto new_size = (size * num) + sizeof(size_t);

    auto *res = ta_alloc(new_size);
    if (!res) {
      std::cerr << "ta_alloc() failed for [" << new_size
                << "] bytes - no memory\n";
      return nullptr;
    }

    memset(res, 0x00, new_size);

    size_t *first_word = reinterpret_cast<size_t *>(res);
    first_word[0] = size;
    return &first_word[1];
  }

  void *realloc_impl(void *ptr, size_t new_size) {
    if (!ptr)
      return alloc_impl(new_size);

    size_t *first_word = reinterpret_cast<size_t *>(ptr);
    first_word--;

    auto old_size = first_word[0];

    if (old_size > new_size)
      old_size = new_size;

    auto *new_area = alloc_impl(new_size);

    if (!new_area) {
      std::cerr << "Failed to allocate new buffer in realloc()\n";
      return nullptr;
    }

    memcpy(new_area, ptr, old_size);

    free_impl(ptr);
    return new_area;
  }

  void free_impl(void *addr) {
    size_t *first_word = reinterpret_cast<size_t *>(addr);
    first_word--;

    ta_free(first_word);
  }

public:
  void *alloc(size_t size) {
    std::unique_lock<std::mutex> lock(_mutex);
    return alloc_impl(size);
  }

  void *calloc(size_t num, size_t size) {
    std::unique_lock<std::mutex> lock(_mutex);
    return calloc_impl(num, size);
  }

  void *realloc(void *ptr, size_t new_size) {
    std::unique_lock<std::mutex> lock(_mutex);
    return realloc_impl(ptr, new_size);
  }

  void free(void *addr) {
    // NOTE: NEVER LOG this function! (like alloc above).
    // This may be called in critical malloc/lib'c heap forbidden paths.
    std::unique_lock<std::mutex> lock(_mutex);
    free_impl(addr);
  }

  static tinyalloc_threadsafe &instance() {
    static tinyalloc_threadsafe s;
    return s;
  }

  tinyalloc_threadsafe() {
    size_t heap_size = 0;
    size_t max_chunks = 0;

    if (const char *heap_env = std::getenv(HEAP_SIZE_ENV)) {
      heap_size = std::strtoull(heap_env, nullptr, 0);
      if (heap_size == ULLONG_MAX || heap_size == 0) {
        std::cerr << "cannot use heap size supplied [" << heap_env << "]\n";
        heap_size = 0;
      }
    }

    if (const char *num_chunks_env = std::getenv(HEAP_NUM_CHUNKS_ENV)) {
      max_chunks = std::strtoull(num_chunks_env, nullptr, 0);
      if (max_chunks == ULLONG_MAX || max_chunks == 0) {
        std::cerr << "cannot use max chunks supplied [" << num_chunks_env
                  << "]\n";
        max_chunks = 0;
      }
    }

    if (!heap_size) {
      heap_size = HEAP_DEFAULT_SIZE;
    }

    if (!max_chunks) {
      max_chunks = HEAP_MAX_NUM_CHUNKS;
    }

    _heap_size = heap_size;

    void *res = mmap(nullptr, _heap_size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (res == MAP_FAILED) {
      int err = errno;
      std::cerr << "mmap() failed for [" << _heap_size << "] bytes, err=["
                << err << "]\n";
      abort();
    }

    _heap = static_cast<uint8_t *>(res);

    __kmp_ns_mark_page(_heap, _heap_size, "openmp tinyalloc heap");

    auto *limit = _heap + heap_size;

    auto *base = _heap;
    if (!ta_init(base, limit, max_chunks, HEAP_SPLIT_THRESHOLD,
                 HEAP_ALIGNMENT)) {
      std::cerr << "failed to initialize tinyalloc heap";
      abort();
    }
  }

  ~tinyalloc_threadsafe() = default;
  tinyalloc_threadsafe(const tinyalloc_threadsafe &) = delete;
  tinyalloc_threadsafe(tinyalloc_threadsafe &&) = delete;
  tinyalloc_threadsafe &operator=(const tinyalloc_threadsafe &) = delete;
  tinyalloc_threadsafe &operator=(tinyalloc_threadsafe &&) = delete;
};

void ensure_initialized() { tinyalloc_threadsafe::instance(); }

void *alloc(size_t size) {
  auto *res = tinyalloc_threadsafe::instance().alloc(size);
  return res;
}

void *calloc(size_t num, size_t size) {
  auto *res = tinyalloc_threadsafe::instance().calloc(num, size);
  return res;
}

void *realloc(void *ptr, size_t new_size) {
  auto *res = tinyalloc_threadsafe::instance().realloc(ptr, new_size);
  return res;
}

void free(void *addr) { tinyalloc_threadsafe::instance().free(addr); }

} // namespace wrapper
} // namespace tinyalloc
} // namespace ns
#endif // LIBOMP_NEXTSILICON_ATOMICS_BYPASS