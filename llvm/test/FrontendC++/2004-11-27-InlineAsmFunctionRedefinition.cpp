// RUN: %llvmgxx %s -S -o /dev/null

// PR397

struct stat { };
struct stat64 { };

extern "C" {

extern int lstat(const char *, struct stat *) __asm__("lstat64");
extern int lstat64(const char *, struct stat64 *);

extern int __lxstat(int, const char *, struct stat *) __asm__("__lxstat64");
extern int __lxstat64(int, const char *, struct stat64 *);

extern __inline__ int lstat(const char *path, struct stat *statbuf) {
    return __lxstat(3, path, statbuf);
}
extern __inline__ int lstat64(const char *path, struct stat64 *statbuf) {
    return __lxstat64(3, path, statbuf);
}
}

int do_one_file(void) {
    return lstat(0, 0) + lstat64(0,0);
}
