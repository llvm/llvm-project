/*
 * DLSymService.h
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef __cplusplus
extern "C" {
#endif
void *get_dlsym_for_name(const char *name);
void *get_library_with_name(const char *name);
const char *get_error();
#ifdef __cplusplus
}
#endif