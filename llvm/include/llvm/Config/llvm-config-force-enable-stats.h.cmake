/*===------- llvm/Config/llvm-config-force-enable-stats.h.cmake ---*- C -*-===*/
/*                                                                            */
/* Part of the LLVM Project, under the Apache License v2.0 with LLVM          */
/* Exceptions.                                                                */
/* See https://llvm.org/LICENSE.txt for license information.                  */
/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    */
/*                                                                            */
/*===----------------------------------------------------------------------===*/

#ifndef LLVM_CONFIG_FORCE_ENABLE_STATS_H
#define LLVM_CONFIG_FORCE_ENABLE_STATS_H

/* Whether LLVM records statistics for use with GetStatistics(),
 * PrintStatistics() or PrintStatisticsJSON()
 */
#cmakedefine01 LLVM_FORCE_ENABLE_STATS

#endif // LLVM_CONFIG_FORCE_ENABLE_STATS_H
