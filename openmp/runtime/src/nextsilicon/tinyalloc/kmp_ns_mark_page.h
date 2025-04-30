#ifndef __MARK_PAGE_HPP__
#define __MARK_PAGE_HPP__

#if LIBOMP_NEXTSILICON_ATOMICS_BYPASS

#include <cstddef>

void __kmp_ns_mark_page(void *addr, size_t size, const char *caller);

#endif // LIBOMP_NEXTSILICON_ATOMICS_BYPASS

#endif // __MARK_PAGE_HPP__