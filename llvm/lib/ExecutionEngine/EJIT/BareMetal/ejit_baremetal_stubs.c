//===-- ejit_baremetal_stubs.c - Bare-metal OS stub library ---------------===//
//
// No-op / error-returning stubs for POSIX functions that the EJIT + LLVM
// stack links against but never calls at runtime on single-threaded
// bare-metal without filesystem.
//
// Self-contained — does NOT include <pthread.h>, <dlfcn.h>, <sys/mman.h> etc.
// because bare-metal toolchains may not provide those.  All types are
// forward-declared as needed.
//
// Link with:  -lejit_baremetal  (or add to firmware link command)
//===----------------------------------------------------------------------===//

#include <errno.h>
#include <stddef.h>

// ── forward declarations (avoid pulling POSIX headers) ─────────────────────

typedef unsigned pthread_mutex_t;
typedef unsigned pthread_mutexattr_t;
typedef unsigned pthread_rwlock_t;
typedef unsigned pthread_once_t;
typedef unsigned sigset_t;
typedef int    pid_t;
typedef long   clockid_t;
typedef long   off_t;
typedef long   ssize_t;
typedef unsigned mode_t;

struct timespec  { long tv_sec; long tv_nsec; };
struct timeval   { long tv_sec; long tv_usec; };
struct stat      { char _[128]; };
struct sigaction { char _[128]; };
struct stack_t   { char _[32]; };

typedef void *(*posix_spawn_file_actions_t);
typedef void *(*posix_spawnattr_t);

// ── errno ──────────────────────────────────────────────────────────────────

int *__errno_location(void) {
  static int _errno;
  return &_errno;
}

// ── pthread ────────────────────────────────────────────────────────────────

int pthread_mutex_lock(pthread_mutex_t *m)          { (void)m; return 0; }
int pthread_mutex_unlock(pthread_mutex_t *m)        { (void)m; return 0; }
int pthread_mutex_init(pthread_mutex_t *m, const pthread_mutexattr_t *a) { (void)m; (void)a; return 0; }
int pthread_mutex_destroy(pthread_mutex_t *m)       { (void)m; return 0; }
int pthread_rwlock_rdlock(pthread_rwlock_t *r)      { (void)r; return 0; }
int pthread_rwlock_wrlock(pthread_rwlock_t *r)      { (void)r; return 0; }
int pthread_rwlock_unlock(pthread_rwlock_t *r)      { (void)r; return 0; }
int pthread_once(pthread_once_t *o, void (*f)(void)) {
  if (!*o) { *o = 1; f(); }
  return 0;
}
int pthread_sigmask(int how, const sigset_t *set, sigset_t *old) {
  (void)how; (void)set; (void)old; return 0;
}

// ── memory ─────────────────────────────────────────────────────────────────

void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
  (void)addr; (void)len; (void)prot; (void)flags; (void)fd; (void)off;
  errno = 12 /*ENOMEM*/;
  return (void *)-1;
}
int munmap(void *addr, size_t len) { (void)addr; (void)len; return 0; }
int mprotect(void *addr, size_t len, int prot) { (void)addr; (void)len; (void)prot; return 0; }

// ── dynamic loading ────────────────────────────────────────────────────────

void *dlopen(const char *f, int flags) { (void)f; (void)flags; return NULL; }
void *dlsym(void *h, const char *s)    { (void)h; (void)s; return NULL; }
int   dlclose(void *h)                 { (void)h; return -1; }
char *dlerror(void) { static char msg[] = "bare-metal: no dylibs"; return msg; }

// ── file I/O ───────────────────────────────────────────────────────────────

int open(const char *p, int f, ...)     { (void)p; (void)f; errno = 2 /*ENOENT*/; return -1; }
int close(int fd)                       { (void)fd; return 0; }
ssize_t read(int fd, void *b, size_t n)   { (void)fd; (void)b; (void)n; errno = 9 /*EBADF*/; return -1; }
ssize_t write(int fd, const void *b, size_t n) { (void)fd; (void)b; (void)n; errno = 9; return -1; }
off_t lseek(int fd, off_t o, int w)     { (void)fd; (void)o; (void)w; errno = 9; return -1; }
int fstat(int fd, void *b)             { (void)fd; (void)b; errno = 9; return -1; }
int stat(const char *p, void *b)       { (void)p; (void)b; errno = 2; return -1; }
int fcntl(int fd, int c, ...)           { (void)fd; (void)c; errno = 9; return -1; }

// ── process ────────────────────────────────────────────────────────────────

pid_t getpid(void)              { return 1; }
char *getenv(const char *n)     { (void)n; return NULL; }
long  sysconf(int name)         { return (name == 30 /*_SC_PAGESIZE*/) ? 4096 : -1; }
pid_t fork(void)                { errno = 38 /*ENOSYS*/; return -1; }
int   execve(const char *p, char *const a[], char *const e[]) { (void)p; (void)a; (void)e; errno = 38; return -1; }
pid_t waitpid(pid_t p, int *s, int o) { (void)p; (void)s; (void)o; errno = 10 /*ECHILD*/; return -1; }

// ── posix_spawn ────────────────────────────────────────────────────────────

int posix_spawn(pid_t *p, const char *f, void *fa, void *attr,
                char *const a[], char *const e[]) {
  (void)p; (void)f; (void)fa; (void)attr; (void)a; (void)e; return 38;
}
int posix_spawn_file_actions_init(void *fa)           { (void)fa; return 0; }
int posix_spawn_file_actions_destroy(void *fa)        { (void)fa; return 0; }
int posix_spawn_file_actions_addopen(void *fa, int fd, const char *p, int f, mode_t m) {
  (void)fa; (void)fd; (void)p; (void)f; (void)m; return 0;
}
int posix_spawn_file_actions_adddup2(void *fa, int fd, int nfd) {
  (void)fa; (void)fd; (void)nfd; return 0;
}

// ── signal ─────────────────────────────────────────────────────────────────

int sigaction(int s, const void *a, void *o) { (void)s; (void)a; (void)o; return 0; }
int sigprocmask(int h, const void *s, void *o) { (void)h; (void)s; (void)o; return 0; }
int sigemptyset(void *s)  { (void)s; return 0; }
int sigfillset(void *s)   { (void)s; return 0; }
int sigaltstack(const void *s, void *o) { (void)s; (void)o; return 0; }
int kill(pid_t p, int s)  { (void)p; (void)s; return 0; }
int raise(int s)          { (void)s; return 0; }

// ── time ───────────────────────────────────────────────────────────────────

typedef long time_t;
time_t time(time_t *t) {
  time_t v = 0;
  if (t) *t = v;
  return v;
}
int clock_gettime(clockid_t c, struct timespec *ts) {
  (void)c; ts->tv_sec = 0; ts->tv_nsec = 0; return 0;
}
int gettimeofday(struct timeval *tv, void *tz) {
  (void)tz; tv->tv_sec = 0; tv->tv_usec = 0; return 0;
}
