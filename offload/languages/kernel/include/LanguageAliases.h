//===------- Aliases.h --- Helpers to make symbol aliases -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE
#error This file should be included, or used, with a LANGUAGE macro set.
#endif

#define MA_IMPL2(PREFIX, L, RTY, NAME, ...)                                    \
  extern "C" [[gnu::alias("__llvm" #NAME)]] RTY PREFIX##L##NAME(__VA_ARGS__);

#define MA_IMPL1(PREFIX, L, RTY, NAME, ...)                                    \
  MA_IMPL2(PREFIX, L, RTY, NAME, __VA_ARGS__)

#define MAKE_ALIAS(PREFIX, RTY, NAME, ...)                                     \
  MA_IMPL1(PREFIX, LANGUAGE, RTY, NAME, __VA_ARGS__)

MAKE_ALIAS(__, void, RegisterFunction, const char *, const char *, char *,
           const char *, int, uint3 *, uint3 *, dim3 *, dim3 *, int *)
MAKE_ALIAS(__, const char *, RegisterFatBinary, const char *)
MAKE_ALIAS(__, void, UnregisterFatBinary, void *)
MAKE_ALIAS(__, void, RegisterVar, void **, char *, char *, const char *, int,
           int, int, int)
MAKE_ALIAS(__, void, RegisterManagedVar, void **, char *, char *, const char *,
           size_t, unsigned)
MAKE_ALIAS(__, void, RegisterSurface, void **, const struct surfaceReference *,
           const void **, const char *, int, int)
MAKE_ALIAS(__, void, RegisterTexture, void **, const struct textureReference *,
           const void **, const char *, int, int, int)

MAKE_ALIAS(__, unsigned, PushCallConfiguration, dim3, dim3, size_t, void *)
MAKE_ALIAS(__, unsigned, PopCallConfiguration, dim3 *, dim3 *, size_t *, void *)

#undef MAKE_ALIAS
#undef MA_IMPL1
#undef MA_IMPL2
