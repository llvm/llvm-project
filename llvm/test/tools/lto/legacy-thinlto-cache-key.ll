; Verify that legacy ThinLTO includes -mllvm arguments in its cache keys.
; Cache an unoutlined object, enable outlining, then reuse the first
; configuration. The products and cache-entry count must match each option set.
; RUN: opt -module-summary -module-hash %s -o %t.o
; RUN: rm -rf %t.cache && mkdir %t.cache

; Populate the cache with an unoutlined object.
; RUN: %ld64 -lto_library %llvmshlibdir/libLTO.dylib -arch x86_64 -macosx_version_min 10.8.0 -dylib -cache_path_lto %t.cache -o %t.not-outlined.dylib %t.o -lSystem
; RUN: llvm-nm --no-llvm-bc %t.not-outlined.dylib | FileCheck --check-prefix=NO-OUTLINED %s
; RUN: ls %t.cache/llvmcache-* | count 1

; Enabling the outliner must miss the existing entry and create a distinct one.
; RUN: %ld64 -lto_library %llvmshlibdir/libLTO.dylib -arch x86_64 -macosx_version_min 10.8.0 -dylib -cache_path_lto %t.cache -o %t.outlined.dylib %t.o -lSystem -mllvm -enable-machine-outliner
; RUN: llvm-nm --no-llvm-bc %t.outlined.dylib | FileCheck --check-prefix=OUTLINED %s
; RUN: ls %t.cache/llvmcache-* | count 2

; Omitting the option again must reuse the unoutlined cache entry.
; RUN: %ld64 -lto_library %llvmshlibdir/libLTO.dylib -arch x86_64 -macosx_version_min 10.8.0 -dylib -cache_path_lto %t.cache -o %t.not-outlined.dylib %t.o -lSystem
; RUN: llvm-nm --no-llvm-bc %t.not-outlined.dylib | FileCheck --check-prefix=NO-OUTLINED %s
; RUN: ls %t.cache/llvmcache-* | count 2

; OUTLINED: OUTLINED_FUNCTION_
; NO-OUTLINED-NOT: OUTLINED_FUNCTION_

target triple = "x86_64-apple-macosx10.8.0"

define void @outline_me() #0 {
  %slot = alloca i32, align 4

  ; Two copies form a profitable outlining candidate.
  store i32 1, ptr %slot, align 4
  store i32 2, ptr %slot, align 4
  store i32 3, ptr %slot, align 4
  store i32 4, ptr %slot, align 4
  store i32 1, ptr %slot, align 4
  store i32 2, ptr %slot, align 4
  store i32 3, ptr %slot, align 4
  store i32 4, ptr %slot, align 4
  ret void
}

; Preserve the stores and make outlining profitable.
attributes #0 = { noinline optnone noredzone "frame-pointer"="all" }
