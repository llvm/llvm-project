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

/// Prints every field of the remark and counts the deliveries in userData.
static void remarkCallback(MlirRemark remark, void *userData) {
  int *count = (int *)userData;
  ++*count;
  const char *kind = "unknown";
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
  mlirRemarkPrint(remark, printToStderr, NULL);
  fprintf(stderr, "\n  location: ");
  mlirLocationPrint(mlirRemarkGetLocation(remark), printToStderr, NULL);
  fprintf(stderr, "\n");
}

static void deleteUserData(void *userData) {
  fprintf(stderr, "deleteUserData called (count=%d)\n", *(int *)userData);
}

static void countRemark(MlirRemark remark, void *userData) {
  (void)remark;
  ++*(int *)userData;
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

  // clang-format off
  // CHECK: remark #1 kind=passed name=Unroll category=Loop full=Loop:inner function=main id=1
  // CHECK:   arg RemarkId=1
  // CHECK:   arg Remark=unrolled
  // CHECK:   arg factor=4
  // CHECK:   print: [Passed] Unroll | Category:Loop:inner | Function=main | Remark=unrolled, RemarkId=1, factor=4
  // CHECK:   location: loc("remarks.c":7:3)
  // CHECK: emitted=1
  // clang-format on
  emitted = emit(loc, MlirRemarkKindPassed, "Unroll", "Loop", "unrolled");
  fprintf(stderr, "emitted=%d\n", emitted);

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

// CHECK-LABEL: @testEmitAsDiagnostics
static void testEmitAsDiagnostics(void) {
  fprintf(stderr, "@testEmitAsDiagnostics\n");
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

  char yamlPath[] = "mlir-capi-remarks-test.yaml";
  MlirLogicalResult res = mlirContextEnableOptimizationRemarksToFile(
      ctx, allCategories(".*"), MlirRemarkPolicyAll, MlirRemarkFileFormatYAML,
      str(yamlPath), /*printAsEmitRemarks=*/false);
  // CHECK: yaml enable ok=1
  fprintf(stderr, "yaml enable ok=%d\n", mlirLogicalResultIsSuccess(res));
  emit(loc, MlirRemarkKindPassed, "Unroll", "Loop", "to file");
  mlirContextFinalizeOptimizationRemarks(ctx);
  FILE *f = fopen(yamlPath, "r");
  bool found = false;
  if (f) {
    char line[256];
    while (fgets(line, sizeof(line), f))
      if (strstr(line, "Unroll"))
        found = true;
    fclose(f);
    remove(yamlPath);
  }
  // CHECK: yaml has remark=1
  fprintf(stderr, "yaml has remark=%d\n", found);

  // The bitstream file starts with the LLVM remark bitstream magic.
  char bitstreamPath[] = "mlir-capi-remarks-test.bitstream";
  res = mlirContextEnableOptimizationRemarksToFile(
      ctx, allCategories(".*"), MlirRemarkPolicyFinal,
      MlirRemarkFileFormatBitstream, str(bitstreamPath),
      /*printAsEmitRemarks=*/false);
  // CHECK: bitstream enable ok=1
  fprintf(stderr, "bitstream enable ok=%d\n", mlirLogicalResultIsSuccess(res));
  emit(loc, MlirRemarkKindPassed, "Unroll", "Loop", "to file");
  mlirContextFinalizeOptimizationRemarks(ctx);
  char magic[5] = {0};
  f = fopen(bitstreamPath, "rb");
  if (f) {
    if (fread(magic, 1, 4, f) != 4)
      magic[0] = '\0';
    fclose(f);
    remove(bitstreamPath);
  }
  // CHECK: bitstream magic=RMRK
  fprintf(stderr, "bitstream magic=%s\n", magic);
  mlirContextDestroy(ctx);
}

// CHECK-LABEL: @testAccessorDefaults
static void testAccessorDefaults(void) {
  fprintf(stderr, "@testAccessorDefaults\n");
  MlirContext ctx = mlirContextCreate();
  MlirLocation loc = mlirLocationUnknownGet(ctx);
  int count = 0;
  MlirLogicalResult res = mlirContextEnableOptimizationRemarksWithCallback(
      ctx, allCategories(".*"), MlirRemarkPolicyAll, remarkCallback, &count,
      NULL, false);
  assert(mlirLogicalResultIsSuccess(res));
  // Unset names fall back to placeholders, the only argument is the id, and
  // null argument arrays are accepted when numArgs is 0.
  // clang-format off
  // CHECK: remark #1 kind=passed name=<unknown remark name> category= full= function=<unknown function> id=1
  // CHECK:   arg RemarkId=1
  // CHECK:   print: [Passed]  | RemarkId=1
  // CHECK:   location: loc(unknown)
  // CHECK: emitted=1
  // clang-format on
  bool emitted =
      mlirEmitOptimizationRemark(loc, MlirRemarkKindPassed, str(""), str(""),
                                 str(""), str(""), str(""), 0, NULL, NULL);
  fprintf(stderr, "emitted=%d\n", emitted);
  // A sub-category alone is the full category name.
  // clang-format off
  // CHECK: remark #2 kind=missed name=Name category= full=Sub function=<unknown function> id=2
  // CHECK:   print: [Missed] Name | Category:Sub | RemarkId=2
  // clang-format on
  mlirEmitOptimizationRemark(loc, MlirRemarkKindMissed, str("Name"), str(""),
                             str("Sub"), str(""), str(""), 0, NULL, NULL);
  // CHECK: unknown emitted=0
  emitted = emit(loc, MlirRemarkKindUnknown, "Name", "Loop", "");
  fprintf(stderr, "unknown emitted=%d\n", emitted);
  mlirContextFinalizeOptimizationRemarks(ctx);
  mlirContextDestroy(ctx);
}

// CHECK-LABEL: @testEnableWhileEnabled
static void testEnableWhileEnabled(void) {
  fprintf(stderr, "@testEnableWhileEnabled\n");
  MlirContext ctx = mlirContextCreate();
  MlirLocation loc = mlirLocationUnknownGet(ctx);
  int first = 0, second = 0;
  MlirLogicalResult res = mlirContextEnableOptimizationRemarksWithCallback(
      ctx, allCategories(".*"), MlirRemarkPolicyAll, countRemark, &first,
      deleteUserData, false);
  // CHECK: first ok=1
  fprintf(stderr, "first ok=%d\n", mlirLogicalResultIsSuccess(res));
  // Every enable variant fails while an engine is active; the callback
  // variant releases the user data it was handed right away.
  // CHECK: deleteUserData called (count=0)
  // CHECK: callback ok=0 file ok=0 plain ok=0
  res = mlirContextEnableOptimizationRemarksWithCallback(
      ctx, allCategories(".*"), MlirRemarkPolicyAll, countRemark, &second,
      deleteUserData, false);
  MlirLogicalResult fileRes = mlirContextEnableOptimizationRemarksToFile(
      ctx, allCategories(".*"), MlirRemarkPolicyAll, MlirRemarkFileFormatYAML,
      str("mlir-capi-remarks-test-unused.yaml"), false);
  MlirLogicalResult plainRes = mlirContextEnableOptimizationRemarks(
      ctx, allCategories(".*"), MlirRemarkPolicyAll, false);
  fprintf(stderr, "callback ok=%d file ok=%d plain ok=%d\n",
          mlirLogicalResultIsSuccess(res), mlirLogicalResultIsSuccess(fileRes),
          mlirLogicalResultIsSuccess(plainRes));
  emit(loc, MlirRemarkKindPassed, "Unroll", "Loop", "");
  // CHECK: first=1 second=0
  fprintf(stderr, "first=%d second=%d\n", first, second);
  // Destroying the context with a live engine releases its user data.
  // CHECK: deleteUserData called (count=1)
  // CHECK: destroyed
  mlirContextDestroy(ctx);
  fprintf(stderr, "destroyed\n");
}

int main(void) {
  testCallbackStreamer();
  testFinalPolicy();
  testEmitAsDiagnostics();
  testFileStreamer();
  testAccessorDefaults();
  testEnableWhileEnabled();
  return 0;
}
