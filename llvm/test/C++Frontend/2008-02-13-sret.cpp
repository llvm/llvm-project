// RUN: %llvmgxx -S -O1 -m32 -emit-llvm %s -o - | grep {store i32} | count 1

// Test that all 8 bytes of ret in check242 are copied, and only 4 bytes of
// ret in check93 are copied (the same LLVM struct is used for both).

typedef __builtin_va_list va_list;
typedef unsigned long size_t;
void *memset(void *, int, size_t);

struct S93 { __attribute__((aligned (8))) void * a; } ;
 extern struct S93 s93;
 struct S93 check93 () { 
  struct S93 ret;
 memset (&ret, 0, sizeof (ret));
 ret.a = s93.a; 
 return ret; }

struct S242 { char * a;int b[1]; } ;
 extern struct S242 s242;

 struct S242 check242 () {
 struct S242 ret;
 memset (&ret, 0, sizeof (ret));
 ret.a = s242.a;
 ret.b[0] = s242.b[0];
 return ret; }

void check93va (int z, ...) { 
 struct S93 arg;
 va_list ap;
 __builtin_va_start(ap,z);
 arg = __builtin_va_arg(ap,struct S93);
  __builtin_va_end(ap); }

void check242va (int z, ...) { 
struct S242 arg;
va_list ap;
__builtin_va_start(ap,z);
 arg = __builtin_va_arg(ap,struct S242);
 __builtin_va_end(ap); }

