#ifndef _EXECINFO_H
#define _EXECINFO_H

#ifdef __cplusplus
extern "C" {
#endif

extern int backtrace (void **__array, int __size);

extern char **backtrace_symbols (void *const *__array, int __size);

extern void backtrace_symbols_fd (void *const *__array, int __size, int __fd);

#ifdef __cplusplus
}
#endif

#endif
