//===- llvm/unittest/Frontend/OpenACCTest.cpp - OpenACC Frontend tests ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Frontend/OpenACC/ACC.h.inc"
#include "gtest/gtest.h"

using namespace llvm;
using namespace acc;

namespace {

static const Clause AllClauses[] = {ACCC_unknown,
                                    ACCC_async,
                                    ACCC_attach,
                                    ACCC_auto,
                                    ACCC_bind,
                                    ACCC_capture,
                                    ACCC_collapse,
                                    ACCC_copy,
                                    ACCC_copyin,
                                    ACCC_copyout,
                                    ACCC_create,
                                    ACCC_default,
                                    ACCC_default_async,
                                    ACCC_delete,
                                    ACCC_detach,
                                    ACCC_device,
                                    ACCC_device_num,
                                    ACCC_deviceptr,
                                    ACCC_device_resident,
                                    ACCC_device_type,
                                    ACCC_finalize,
                                    ACCC_firstprivate,
                                    ACCC_gang,
                                    ACCC_host,
                                    ACCC_if,
                                    ACCC_if_present,
                                    ACCC_independent,
                                    ACCC_link,
                                    ACCC_no_create,
                                    ACCC_nohost,
                                    ACCC_num_gangs,
                                    ACCC_num_workers,
                                    ACCC_present,
                                    ACCC_private,
                                    ACCC_read,
                                    ACCC_reduction,
                                    ACCC_self,
                                    ACCC_seq,
                                    ACCC_tile,
                                    ACCC_unknown,
                                    ACCC_use_device,
                                    ACCC_vector,
                                    ACCC_vector_length,
                                    ACCC_wait,
                                    ACCC_worker,
                                    ACCC_write};

TEST(OpenACCTest, DirectiveHelpers) {
  EXPECT_EQ(getOpenACCDirectiveKind(""), ACCD_unknown);
  EXPECT_EQ(getOpenACCDirectiveKind("dummy"), ACCD_unknown);
  EXPECT_EQ(getOpenACCDirectiveKind("atomic"), ACCD_atomic);
  EXPECT_EQ(getOpenACCDirectiveKind("cache"), ACCD_cache);
  EXPECT_EQ(getOpenACCDirectiveKind("data"), ACCD_data);
  EXPECT_EQ(getOpenACCDirectiveKind("declare"), ACCD_declare);
  EXPECT_EQ(getOpenACCDirectiveKind("enter data"), ACCD_enter_data);
  EXPECT_EQ(getOpenACCDirectiveKind("exit data"), ACCD_exit_data);
  EXPECT_EQ(getOpenACCDirectiveKind("host_data"), ACCD_host_data);
  EXPECT_EQ(getOpenACCDirectiveKind("init"), ACCD_init);
  EXPECT_EQ(getOpenACCDirectiveKind("kernels"), ACCD_kernels);
  EXPECT_EQ(getOpenACCDirectiveKind("kernels loop"), ACCD_kernels_loop);
  EXPECT_EQ(getOpenACCDirectiveKind("loop"), ACCD_loop);
  EXPECT_EQ(getOpenACCDirectiveKind("parallel"), ACCD_parallel);
  EXPECT_EQ(getOpenACCDirectiveKind("parallel loop"), ACCD_parallel_loop);
  EXPECT_EQ(getOpenACCDirectiveKind("routine"), ACCD_routine);
  EXPECT_EQ(getOpenACCDirectiveKind("serial"), ACCD_serial);
  EXPECT_EQ(getOpenACCDirectiveKind("serial loop"), ACCD_serial_loop);
  EXPECT_EQ(getOpenACCDirectiveKind("set"), ACCD_set);
  EXPECT_EQ(getOpenACCDirectiveKind("shutdown"), ACCD_shutdown);
  EXPECT_EQ(getOpenACCDirectiveKind("unknown"), ACCD_unknown);
  EXPECT_EQ(getOpenACCDirectiveKind("update"), ACCD_update);
  EXPECT_EQ(getOpenACCDirectiveKind("wait"), ACCD_wait);

  EXPECT_EQ(getOpenACCDirectiveName(ACCD_atomic), "atomic");
  EXPECT_EQ(getOpenACCDirectiveName(ACCD_cache), "cache");
  EXPECT_EQ(getOpenACCDirectiveName(ACCD_data), "data");
  EXPECT_EQ(getOpenACCDirectiveName(ACCD_declare), "declare");
  EXPECT_EQ(getOpenACCDirectiveName(ACCD_enter_data), "enter data");
  EXPECT_EQ(getOpenACCDirectiveName(ACCD_exit_data), "exit data");
  EXPECT_EQ(getOpenACCDirectiveName(ACCD_host_data), "host_data");
  EXPECT_EQ(getOpenACCDirectiveName(ACCD_init), "init");
  EXPECT_EQ(getOpenACCDirectiveName(ACCD_kernels), "kernels");
  EXPECT_EQ(getOpenACCDirectiveName(ACCD_kernels_loop), "kernels loop");
  EXPECT_EQ(getOpenACCDirectiveName(ACCD_loop), "loop");
  EXPECT_EQ(getOpenACCDirectiveName(ACCD_parallel), "parallel");
  EXPECT_EQ(getOpenACCDirectiveName(ACCD_parallel_loop), "parallel loop");
  EXPECT_EQ(getOpenACCDirectiveName(ACCD_routine), "routine");
  EXPECT_EQ(getOpenACCDirectiveName(ACCD_serial), "serial");
  EXPECT_EQ(getOpenACCDirectiveName(ACCD_serial_loop), "serial loop");
  EXPECT_EQ(getOpenACCDirectiveName(ACCD_set), "set");
  EXPECT_EQ(getOpenACCDirectiveName(ACCD_shutdown), "shutdown");
  EXPECT_EQ(getOpenACCDirectiveName(ACCD_unknown), "unknown");
  EXPECT_EQ(getOpenACCDirectiveName(ACCD_update), "update");
  EXPECT_EQ(getOpenACCDirectiveName(ACCD_wait), "wait");
}

TEST(OpenACCTest, ClauseHelpers) {
  EXPECT_EQ(getOpenACCClauseKind(""), ACCC_unknown);
  EXPECT_EQ(getOpenACCClauseKind("dummy"), ACCC_unknown);
  EXPECT_EQ(getOpenACCClauseKind("async"), ACCC_async);
  EXPECT_EQ(getOpenACCClauseKind("attach"), ACCC_attach);
  EXPECT_EQ(getOpenACCClauseKind("auto"), ACCC_auto);
  EXPECT_EQ(getOpenACCClauseKind("bind"), ACCC_bind);
  EXPECT_EQ(getOpenACCClauseKind("capture"), ACCC_capture);
  EXPECT_EQ(getOpenACCClauseKind("collapse"), ACCC_collapse);
  EXPECT_EQ(getOpenACCClauseKind("copy"), ACCC_copy);
  EXPECT_EQ(getOpenACCClauseKind("copyin"), ACCC_copyin);
  EXPECT_EQ(getOpenACCClauseKind("copyout"), ACCC_copyout);
  EXPECT_EQ(getOpenACCClauseKind("create"), ACCC_create);
  EXPECT_EQ(getOpenACCClauseKind("default"), ACCC_default);
  EXPECT_EQ(getOpenACCClauseKind("default_async"), ACCC_default_async);
  EXPECT_EQ(getOpenACCClauseKind("delete"), ACCC_delete);
  EXPECT_EQ(getOpenACCClauseKind("detach"), ACCC_detach);
  EXPECT_EQ(getOpenACCClauseKind("device"), ACCC_device);
  EXPECT_EQ(getOpenACCClauseKind("device_num"), ACCC_device_num);
  EXPECT_EQ(getOpenACCClauseKind("deviceptr"), ACCC_deviceptr);
  EXPECT_EQ(getOpenACCClauseKind("device_resident"), ACCC_device_resident);
  EXPECT_EQ(getOpenACCClauseKind("device_type"), ACCC_device_type);
  EXPECT_EQ(getOpenACCClauseKind("finalize"), ACCC_finalize);
  EXPECT_EQ(getOpenACCClauseKind("firstprivate"), ACCC_firstprivate);
  EXPECT_EQ(getOpenACCClauseKind("gang"), ACCC_gang);
  EXPECT_EQ(getOpenACCClauseKind("host"), ACCC_host);
  EXPECT_EQ(getOpenACCClauseKind("if"), ACCC_if);
  EXPECT_EQ(getOpenACCClauseKind("if_present"), ACCC_if_present);
  EXPECT_EQ(getOpenACCClauseKind("independent"), ACCC_independent);
  EXPECT_EQ(getOpenACCClauseKind("link"), ACCC_link);
  EXPECT_EQ(getOpenACCClauseKind("no_create"), ACCC_no_create);
  EXPECT_EQ(getOpenACCClauseKind("nohost"), ACCC_nohost);
  EXPECT_EQ(getOpenACCClauseKind("num_gangs"), ACCC_num_gangs);
  EXPECT_EQ(getOpenACCClauseKind("num_workers"), ACCC_num_workers);
  EXPECT_EQ(getOpenACCClauseKind("present"), ACCC_present);
  EXPECT_EQ(getOpenACCClauseKind("private"), ACCC_private);
  EXPECT_EQ(getOpenACCClauseKind("read"), ACCC_read);
  EXPECT_EQ(getOpenACCClauseKind("reduction"), ACCC_reduction);
  EXPECT_EQ(getOpenACCClauseKind("self"), ACCC_self);
  EXPECT_EQ(getOpenACCClauseKind("seq"), ACCC_seq);
  EXPECT_EQ(getOpenACCClauseKind("tile"), ACCC_tile);
  EXPECT_EQ(getOpenACCClauseKind("unknown"), ACCC_unknown);
  EXPECT_EQ(getOpenACCClauseKind("use_device"), ACCC_use_device);
  EXPECT_EQ(getOpenACCClauseKind("vector"), ACCC_vector);
  EXPECT_EQ(getOpenACCClauseKind("vector_length"), ACCC_vector_length);
  EXPECT_EQ(getOpenACCClauseKind("wait"), ACCC_wait);
  EXPECT_EQ(getOpenACCClauseKind("worker"), ACCC_worker);
  EXPECT_EQ(getOpenACCClauseKind("write"), ACCC_write);

  EXPECT_EQ(getOpenACCClauseName(ACCC_async), "async");
  EXPECT_EQ(getOpenACCClauseName(ACCC_attach), "attach");
  EXPECT_EQ(getOpenACCClauseName(ACCC_auto), "auto");
  EXPECT_EQ(getOpenACCClauseName(ACCC_bind), "bind");
  EXPECT_EQ(getOpenACCClauseName(ACCC_capture), "capture");
  EXPECT_EQ(getOpenACCClauseName(ACCC_collapse), "collapse");
  EXPECT_EQ(getOpenACCClauseName(ACCC_copy), "copy");
  EXPECT_EQ(getOpenACCClauseName(ACCC_copyin), "copyin");
  EXPECT_EQ(getOpenACCClauseName(ACCC_copyout), "copyout");
  EXPECT_EQ(getOpenACCClauseName(ACCC_create), "create");
  EXPECT_EQ(getOpenACCClauseName(ACCC_default), "default");
  EXPECT_EQ(getOpenACCClauseName(ACCC_default_async), "default_async");
  EXPECT_EQ(getOpenACCClauseName(ACCC_delete), "delete");
  EXPECT_EQ(getOpenACCClauseName(ACCC_detach), "detach");
  EXPECT_EQ(getOpenACCClauseName(ACCC_device), "device");
  EXPECT_EQ(getOpenACCClauseName(ACCC_device_num), "device_num");
  EXPECT_EQ(getOpenACCClauseName(ACCC_deviceptr), "deviceptr");
  EXPECT_EQ(getOpenACCClauseName(ACCC_device_resident), "device_resident");
  EXPECT_EQ(getOpenACCClauseName(ACCC_device_type), "device_type");
  EXPECT_EQ(getOpenACCClauseName(ACCC_finalize), "finalize");
  EXPECT_EQ(getOpenACCClauseName(ACCC_firstprivate), "firstprivate");
  EXPECT_EQ(getOpenACCClauseName(ACCC_gang), "gang");
  EXPECT_EQ(getOpenACCClauseName(ACCC_host), "host");
  EXPECT_EQ(getOpenACCClauseName(ACCC_if), "if");
  EXPECT_EQ(getOpenACCClauseName(ACCC_if_present), "if_present");
  EXPECT_EQ(getOpenACCClauseName(ACCC_independent), "independent");
  EXPECT_EQ(getOpenACCClauseName(ACCC_link), "link");
  EXPECT_EQ(getOpenACCClauseName(ACCC_no_create), "no_create");
  EXPECT_EQ(getOpenACCClauseName(ACCC_nohost), "nohost");
  EXPECT_EQ(getOpenACCClauseName(ACCC_num_gangs), "num_gangs");
  EXPECT_EQ(getOpenACCClauseName(ACCC_num_workers), "num_workers");
  EXPECT_EQ(getOpenACCClauseName(ACCC_present), "present");
  EXPECT_EQ(getOpenACCClauseName(ACCC_private), "private");
  EXPECT_EQ(getOpenACCClauseName(ACCC_read), "read");
  EXPECT_EQ(getOpenACCClauseName(ACCC_reduction), "reduction");
  EXPECT_EQ(getOpenACCClauseName(ACCC_self), "self");
  EXPECT_EQ(getOpenACCClauseName(ACCC_seq), "seq");
  EXPECT_EQ(getOpenACCClauseName(ACCC_tile), "tile");
  EXPECT_EQ(getOpenACCClauseName(ACCC_unknown), "unknown");
  EXPECT_EQ(getOpenACCClauseName(ACCC_use_device), "use_device");
  EXPECT_EQ(getOpenACCClauseName(ACCC_vector), "vector");
  EXPECT_EQ(getOpenACCClauseName(ACCC_vector_length), "vector_length");
  EXPECT_EQ(getOpenACCClauseName(ACCC_wait), "wait");
  EXPECT_EQ(getOpenACCClauseName(ACCC_worker), "worker");
  EXPECT_EQ(getOpenACCClauseName(ACCC_write), "write");
}

static void expectAllowedClauses(Directive Dir, unsigned Version,
                                 const ArrayRef<Clause> &AllowedClauses) {
  SmallSet<Clause, 30> AllowedClausesSet;
  for (Clause Cl : AllowedClauses) {
    EXPECT_TRUE(isAllowedClauseForDirective(Dir, Cl, Version));
    AllowedClausesSet.insert(Cl);
  }

  for (Clause Cl : AllClauses) {
    if (!AllowedClausesSet.contains(Cl)) {
      EXPECT_FALSE(isAllowedClauseForDirective(Dir, Cl, Version));
    }
  }
}

TEST(OpenACCTest, AllowedClause) {
  expectAllowedClauses(ACCD_atomic, 3, {});
  expectAllowedClauses(ACCD_cache, 3, {});
  expectAllowedClauses(ACCD_unknown, 3, {});
  expectAllowedClauses(ACCD_parallel, 0, {}); // Version starts at 1

  expectAllowedClauses(ACCD_data, 3,
                       {ACCC_if, ACCC_attach, ACCC_copy, ACCC_copyin,
                        ACCC_copyout, ACCC_create, ACCC_default, ACCC_deviceptr,
                        ACCC_no_create, ACCC_present});

  expectAllowedClauses(ACCD_declare, 3,
                       {ACCC_copy, ACCC_copyin, ACCC_copyout, ACCC_create,
                        ACCC_present, ACCC_deviceptr, ACCC_device_resident,
                        ACCC_link});

  expectAllowedClauses(
      ACCD_enter_data, 3,
      {ACCC_async, ACCC_if, ACCC_wait, ACCC_attach, ACCC_create, ACCC_copyin});

  expectAllowedClauses(ACCD_exit_data, 3,
                       {ACCC_async, ACCC_if, ACCC_wait, ACCC_finalize,
                        ACCC_copyout, ACCC_delete, ACCC_detach});

  expectAllowedClauses(ACCD_host_data, 3,
                       {ACCC_if, ACCC_if_present, ACCC_use_device});

  expectAllowedClauses(ACCD_init, 3,
                       {ACCC_device_num, ACCC_device_type, ACCC_if});

  expectAllowedClauses(ACCD_kernels, 3,
                       {ACCC_attach, ACCC_copy, ACCC_copyin, ACCC_copyout,
                        ACCC_create, ACCC_device_type, ACCC_no_create,
                        ACCC_present, ACCC_deviceptr, ACCC_async, ACCC_default,
                        ACCC_if, ACCC_num_gangs, ACCC_num_workers, ACCC_self,
                        ACCC_vector_length, ACCC_wait});

  expectAllowedClauses(
      ACCD_kernels_loop, 3,
      {ACCC_copy,        ACCC_copyin,        ACCC_copyout, ACCC_create,
       ACCC_device_type, ACCC_no_create,     ACCC_present, ACCC_private,
       ACCC_deviceptr,   ACCC_attach,        ACCC_async,   ACCC_collapse,
       ACCC_default,     ACCC_gang,          ACCC_if,      ACCC_num_gangs,
       ACCC_num_workers, ACCC_reduction,     ACCC_self,    ACCC_tile,
       ACCC_vector,      ACCC_vector_length, ACCC_wait,    ACCC_worker,
       ACCC_auto,        ACCC_independent,   ACCC_seq});

  expectAllowedClauses(ACCD_loop, 3,
                       {ACCC_device_type, ACCC_private, ACCC_collapse,
                        ACCC_gang, ACCC_reduction, ACCC_tile, ACCC_vector,
                        ACCC_worker, ACCC_auto, ACCC_independent, ACCC_seq});

  expectAllowedClauses(ACCD_parallel, 3,
                       {ACCC_async,        ACCC_wait,          ACCC_num_gangs,
                        ACCC_num_workers,  ACCC_vector_length, ACCC_device_type,
                        ACCC_if,           ACCC_self,          ACCC_reduction,
                        ACCC_copy,         ACCC_copyin,        ACCC_copyout,
                        ACCC_create,       ACCC_no_create,     ACCC_present,
                        ACCC_deviceptr,    ACCC_attach,        ACCC_private,
                        ACCC_firstprivate, ACCC_default});

  expectAllowedClauses(
      ACCD_parallel_loop, 3,
      {ACCC_attach,    ACCC_copy,      ACCC_copyin,      ACCC_copyout,
       ACCC_create,    ACCC_deviceptr, ACCC_device_type, ACCC_firstprivate,
       ACCC_no_create, ACCC_present,   ACCC_private,     ACCC_tile,
       ACCC_wait,      ACCC_async,     ACCC_collapse,    ACCC_default,
       ACCC_gang,      ACCC_if,        ACCC_num_gangs,   ACCC_num_workers,
       ACCC_reduction, ACCC_self,      ACCC_vector,      ACCC_vector_length,
       ACCC_worker,    ACCC_auto,      ACCC_independent, ACCC_seq});

  expectAllowedClauses(ACCD_routine, 3,
                       {ACCC_bind, ACCC_device_type, ACCC_nohost, ACCC_gang,
                        ACCC_seq, ACCC_vector, ACCC_worker});

  expectAllowedClauses(ACCD_serial, 3,
                       {ACCC_attach, ACCC_copy, ACCC_copyin, ACCC_copyout,
                        ACCC_create, ACCC_deviceptr, ACCC_device_type,
                        ACCC_firstprivate, ACCC_no_create, ACCC_present,
                        ACCC_private, ACCC_wait, ACCC_async, ACCC_default,
                        ACCC_if, ACCC_reduction, ACCC_self});

  expectAllowedClauses(
      ACCD_serial_loop, 3,
      {ACCC_attach,    ACCC_copy,      ACCC_copyin,      ACCC_copyout,
       ACCC_create,    ACCC_deviceptr, ACCC_device_type, ACCC_firstprivate,
       ACCC_no_create, ACCC_present,   ACCC_private,     ACCC_wait,
       ACCC_async,     ACCC_collapse,  ACCC_default,     ACCC_gang,
       ACCC_if,        ACCC_reduction, ACCC_self,        ACCC_tile,
       ACCC_vector,    ACCC_worker,    ACCC_auto,        ACCC_independent,
       ACCC_seq});

  expectAllowedClauses(
      ACCD_set, 3,
      {ACCC_default_async, ACCC_device_num, ACCC_device_type, ACCC_if});

  expectAllowedClauses(ACCD_shutdown, 3,
                       {ACCC_device_num, ACCC_device_type, ACCC_if});

  expectAllowedClauses(ACCD_update, 3,
                       {ACCC_async, ACCC_wait, ACCC_device_type, ACCC_if,
                        ACCC_if_present, ACCC_self, ACCC_host, ACCC_device});

  expectAllowedClauses(ACCD_wait, 3, {ACCC_async, ACCC_if});
}

} // namespace
