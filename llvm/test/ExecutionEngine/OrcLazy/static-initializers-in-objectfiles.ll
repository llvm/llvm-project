; RUN: rm -rf %t
; RUN: mkdir -p %t
; RUN: lli -jit-kind=orc-lazy -enable-cache-manager -object-cache-dir=%t %s
; RUN: lli -jit-kind=orc-lazy -enable-cache-manager -object-cache-dir=%t %s
;
; Verify that LLJIT Platforms respect static initializers in cached objects.
; This IR file contains a static initializer that must execute for main to exit
; with value zero. The first execution will populate an object cache for the
; second. The initializer in the cached objects must also be successfully run
; for the test to pass.

@HasError = global i8 1, align 1
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @resetHasError, i8* null }]

define void @resetHasError() {
entry:
  store i8 0, i8* @HasError, align 1
  ret void
}

define i32 @main(i32 %argc, i8** %argv) #2 {
entry:
  %0 = load i8, i8* @HasError, align 1
  %tobool = trunc i8 %0 to i1
  %conv = zext i1 %tobool to i32
  ret i32 %conv
}

