#ifndef __AT_EXIT_H__
#define __AT_EXIT_H__

typedef void __atexit_func(void *);

extern int __cxa_atexit(__atexit_func *, void *, void *) __attribute__((weak));

#endif /* __AT_EXIT_H__ */
