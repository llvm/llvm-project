//===- remarks.c - Test of the remark engine C API ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/* RUN: mlir-capi-remarks-test 2>&1 | FileCheck %s
 */

#include "mlir-c/Remarks.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void printToStderr(MlirStringRef str, void *userData) {
  (void)userData;
  fwrite(str.data, 1, str.length, stderr);
}

static MlirStringRef str(const char *s) {
  return mlirStringRefCreateFromCString(s);
}

static MlirRemarkCategories allCategories(const char *filter) {
  MlirRemarkCategories cats;
  memset(&cats, 0, sizeof(cats));
  cats.all = str(filter);
  return cats;
}

/// The callback streamer: prints every field of the remark it receives and
/// counts the deliveries in userData.
static void remarkCallback(MlirRemark remark, void *userData) {
  int *count = (int *)userData;
  ++*count;
  const char *kind = "?";
  switch (mlirRemarkGetKind(remark)) {
  case MlirRemarkKindPassed:
    kind = "passed";
    break;
  case MlirRemarkKindMissed:
    kind = "missed";
    break;
  case MlirRemarkKindFailure:
    kind = "failure";
    break;
  case MlirRemarkKindAnalysis:
    kind = "analysis";
    break;
  case MlirRemarkKindUnknown:
    kind = "unknown";
    break;
  }
  MlirStringRef name = mlirRemarkGetRemarkName(remark);
  MlirStringRef category = mlirRemarkGetCategoryName(remark);
  MlirStringRef fullCategory = mlirRemarkGetFullCategoryName(remark);
  MlirStringRef function = mlirRemarkGetFunctionName(remark);
  fprintf(stderr,
          "remark #%d kind=%s name=%.*s category=%.*s full=%.*s "
          "function=%.*s id=%" PRIu64 "\n",
          *count, kind, (int)name.length, name.data, (int)category.length,
          category.data, (int)fullCategory.length, fullCategory.data,
          (int)function.length, function.data, mlirRemarkGetId(remark));
  intptr_t numArgs = mlirRemarkGetNumArgs(remark);
  for (intptr_t i = 0; i < numArgs; ++i) {
    MlirStringRef key = mlirRemarkGetArgKey(remark, i);
    MlirStringRef value = mlirRemarkGetArgValue(remark, i);
    fprintf(stderr, "  arg %.*s=%.*s\n", (int)key.length, key.data,
            (int)value.length, value.data);
  }
  fprintf(stderr, "  print: ");
  mlirRemarkPrint(remark, /*printLocation=*/true, printToStderr, NULL);
  fprintf(stderr, "\n  location: ");
  mlirLocationPrint(mlirRemarkGetLocation(remark), printToStderr, NULL);
  fprintf(stderr, "\n");
}

static void deleteUserData(void *userData) {
  fprintf(stderr, "deleteUserData called (count=%d)\n", *(int *)userData);
}

static bool emit(MlirLocation loc, MlirRemarkKind kind, const char *name,
                 const char *category, const char *message) {
  MlirStringRef keys[1] = {str("factor")};
  MlirStringRef values[1] = {str("4")};
  return mlirEmitOptimizationRemark(loc, kind, str(name), str(category),
                                    str("inner"), str("main"), str(message), 1,
                                    keys, values);
}

// CHECK-LABEL: @testCallbackStreamer
static void testCallbackStreamer(void) {
  fprintf(stderr, "@testCallbackStreamer\n");
  MlirContext ctx = mlirContextCreate();
  MlirLocation loc = mlirLocationFileLineColGet(ctx, str("remarks.c"), 7, 3);

  // Without an engine nothing is emitted.
  // CHECK: no engine: enabled=0 emitted=0
  bool emitted = emit(loc, MlirRemarkKindPassed, "Unroll", "Loop", "dropped");
  fprintf(stderr, "no engine: enabled=%d emitted=%d\n",
          mlirContextHasRemarkEngine(ctx), emitted);

  int count = 0;
  MlirLogicalResult res = mlirContextEnableOptimizationRemarksWithCallback(
      ctx, allCategories("Loop"), MlirRemarkPolicyAll, remarkCallback, &count,
      deleteUserData, /*printAsEmitRemarks=*/false);
  // CHECK: enabled=1 ok=1
  fprintf(stderr, "enabled=%d ok=%d\n", mlirContextHasRemarkEngine(ctx),
          mlirLogicalResultIsSuccess(res));

  // Enabling twice fails and leaves the first engine in place.
  // CHECK: second enable ok=0
  res = mlirContextEnableOptimizationRemarks(ctx, allCategories(".*"),
                                             MlirRemarkPolicyAll, true);
  fprintf(stderr, "second enable ok=%d\n", mlirLogicalResultIsSuccess(res));

  // clang-format off
  // CHECK: remark #1 kind=passed name=Unroll category=Loop full=Loop:inner function=main id=1
  // CHECK:   arg RemarkId=1
  // CHECK:   arg Remark=unrolled
  // CHECK:   arg factor=4
  // CHECK:   print: [Passed] Unroll | Category:Loop:inner | Function=main |  @"remarks.c":7:3{{ ?}}Remark=unrolled, RemarkId=1, factor=4
  // CHECK:   location: loc("remarks.c":7:3)
  // CHECK: emitted=1
  // clang-format on
  emitted = emit(loc, MlirRemarkKindPassed, "Unroll", "Loop", "unrolled");
  fprintf(stderr, "emitted=%d\n", emitted);

  // A category the filter does not match is dropped before the streamer.
  // CHECK-NOT: name=Vectorize
  // CHECK: count after filtered emit=1
  emit(loc, MlirRemarkKindMissed, "Vectorize", "Vector", "not matched");
  fprintf(stderr, "count after filtered emit=%d\n", count);

  // CHECK: remark #2 kind=missed name=Interchange
  // CHECK: remark #3 kind=failure name=Fuse
  // CHECK: remark #4 kind=analysis name=TripCount
  emit(loc, MlirRemarkKindMissed, "Interchange", "Loop", "no");
  emit(loc, MlirRemarkKindFailure, "Fuse", "Loop", "no");
  emit(loc, MlirRemarkKindAnalysis, "TripCount", "Loop", "4");

  // Finalize drops the engine and releases the user data.
  // CHECK: deleteUserData called (count=4)
  // CHECK: after finalize: enabled=0 count=4
  mlirContextFinalizeOptimizationRemarks(ctx);
  fprintf(stderr, "after finalize: enabled=%d count=%d\n",
          mlirContextHasRemarkEngine(ctx), count);
  // Finalizing without an engine is a no-op.
  mlirContextFinalizeOptimizationRemarks(ctx);
  mlirContextDestroy(ctx);
}

// CHECK-LABEL: @testFinalPolicy
static void testFinalPolicy(void) {
  fprintf(stderr, "@testFinalPolicy\n");
  MlirContext ctx = mlirContextCreate();
  MlirLocation loc = mlirLocationUnknownGet(ctx);
  int count = 0;
  MlirRemarkCategories cats;
  memset(&cats, 0, sizeof(cats));
  cats.passed = str("Loop");
  MlirLogicalResult res = mlirContextEnableOptimizationRemarksWithCallback(
      ctx, cats, MlirRemarkPolicyFinal, remarkCallback, &count, NULL, false);
  assert(mlirLogicalResultIsSuccess(res));
  emit(loc, MlirRemarkKindPassed, "Unroll", "Loop", "postponed");
  // Only the passed filter is set: a missed remark is not reported.
  emit(loc, MlirRemarkKindMissed, "Interchange", "Loop", "dropped");
  // CHECK: before finalize count=0
  fprintf(stderr, "before finalize count=%d\n", count);
  // CHECK: remark #1 kind=passed name=Unroll
  // CHECK: after finalize count=1
  mlirContextFinalizeOptimizationRemarks(ctx);
  fprintf(stderr, "after finalize count=%d\n", count);
  mlirContextDestroy(ctx);
}

static MlirLogicalResult diagnosticHandler(MlirDiagnostic diagnostic,
                                           void *userData) {
  (void)userData;
  const char *severity = "other";
  if (mlirDiagnosticGetSeverity(diagnostic) == MlirDiagnosticRemark)
    severity = "remark";
  fprintf(stderr, "diagnostic %s: ", severity);
  mlirDiagnosticPrint(diagnostic, printToStderr, NULL);
  fprintf(stderr, "\n");
  return mlirLogicalResultSuccess();
}

// CHECK-LABEL: @testEmitRemarkForm
static void testEmitRemarkForm(void) {
  fprintf(stderr, "@testEmitRemarkForm\n");
  MlirContext ctx = mlirContextCreate();
  MlirLocation loc = mlirLocationUnknownGet(ctx);
  MlirDiagnosticHandlerID id =
      mlirContextAttachDiagnosticHandler(ctx, diagnosticHandler, NULL, NULL);
  MlirLogicalResult res = mlirContextEnableOptimizationRemarks(
      ctx, allCategories(".*"), MlirRemarkPolicyAll,
      /*printAsEmitRemarks=*/true);
  assert(mlirLogicalResultIsSuccess(res));
  // clang-format off
  // CHECK: diagnostic remark: [Analysis] TripCount | Category:Loop:inner | Function=main | Remark=4, RemarkId=1, factor=4
  // clang-format on
  emit(loc, MlirRemarkKindAnalysis, "TripCount", "Loop", "4");
  mlirContextFinalizeOptimizationRemarks(ctx);
  mlirContextDetachDiagnosticHandler(ctx, id);
  mlirContextDestroy(ctx);
}

// CHECK-LABEL: @testFileStreamer
static void testFileStreamer(void) {
  fprintf(stderr, "@testFileStreamer\n");
  MlirContext ctx = mlirContextCreate();
  MlirLocation loc = mlirLocationUnknownGet(ctx);
  char path[] = "mlir-capi-remarks-test.yaml";
  MlirLogicalResult res = mlirContextEnableOptimizationRemarksToFile(
      ctx, allCategories(".*"), MlirRemarkPolicyAll, MlirRemarkFileFormatYAML,
      str(path), false);
  // CHECK: file enable ok=1
  fprintf(stderr, "file enable ok=%d\n", mlirLogicalResultIsSuccess(res));
  emit(loc, MlirRemarkKindPassed, "Unroll", "Loop", "to file");
  mlirContextFinalizeOptimizationRemarks(ctx);
  FILE *f = fopen(path, "r");
  bool found = false;
  if (f) {
    char line[256];
    while (fgets(line, sizeof(line), f))
      if (strstr(line, "Unroll"))
        found = true;
    fclose(f);
    remove(path);
  }
  // CHECK: yaml has remark=1
  fprintf(stderr, "yaml has remark=%d\n", found);
  mlirContextDestroy(ctx);
}

int main(void) {
  testCallbackStreamer();
  testFinalPolicy();
  testEmitRemarkForm();
  testFileStreamer();
  return 0;
}
