// Verifies that --mlir-private-passes + --mlir-private-name-obfuscator
// obfuscates the argument and name of every pass, empties out the pass /
// per-option / per-statistic descriptions, and keeps emitting the
// registration / C-API helpers under each pass's (unchanged) C++ class
// name.
//
// RUN: mlir-tblgen -gen-pass-decls -name=Test -I %S/../../include %s \
// RUN:   | FileCheck %s --check-prefix=BASELINE
// RUN: mlir-tblgen -gen-pass-decls -name=Test -I %S/../../include %s \
// RUN:     --mlir-private-name-obfuscator=md5sum --mlir-private-passes \
// RUN:   | FileCheck %s --check-prefix=OBFUSCATED
// RUN: mlir-tblgen -gen-pass-capi-header -prefix=Test -I %S/../../include %s \
// RUN:     --mlir-private-name-obfuscator=md5sum --mlir-private-passes \
// RUN:   | FileCheck %s --check-prefix=CAPI

include "mlir/Pass/PassBase.td"

def FirstTestPass : Pass<"first-test-pass"> {
  let summary = "First pass summary";
  let options = [
    Option<"opt", "first-opt", "int", "0", "First option description.">
  ];
}

def SecondTestPass : Pass<"second-test-pass"> {
  let summary = "Second pass summary";
}

// Per-pass blocks emerge in alphabetical order, and registration helpers
// follow the same order.
//
// The obfuscated forms used below are the md5 hashes of the original
// names, prefixed with `_`:
//
//   printf %s first-test-pass  | md5sum  -> 21ebd1b31ed5dbc81e5aa9e2fed732f3
//   printf %s FirstTestPass    | md5sum  -> 6f7c29e76fd9f6f176504b48059b6b8b
//   printf %s second-test-pass | md5sum  -> ba8fce64443d8c815b0ed7477793587c
//   printf %s SecondTestPass   | md5sum  -> 6b8d928968d2a98e9bc5af1bffb90f20
//
// Baseline (no flags): names and descriptions are emitted verbatim, both
// passes are registered.
// BASELINE: getArgument() const override { return "first-test-pass"; }
// BASELINE: getDescription() const override { return R"PD(First pass summary)PD"; }
// BASELINE: getName() const override { return "FirstTestPass"; }
// BASELINE: ::llvm::cl::desc(R"PO(First option description.)PO")
// BASELINE: getArgument() const override { return "second-test-pass"; }
// BASELINE: getDescription() const override { return R"PD(Second pass summary)PD"; }
// BASELINE: getName() const override { return "SecondTestPass"; }
// BASELINE: inline void registerFirstTestPass()
// BASELINE: inline void registerSecondTestPass()

// Obfuscated: every pass argument and name is hashed; every description,
// per-option `cl::desc`, and (per the BASELINE list) summary text is
// emitted as the empty string. The C++ class names -- and the per-pass
// registration helpers that use them -- are unchanged.
// OBFUSCATED: getArgument() const override { return "_21ebd1b31ed5dbc81e5aa9e2fed732f3"; }
// OBFUSCATED: getDescription() const override { return R"PD()PD"; }
// OBFUSCATED: getName() const override { return "_6f7c29e76fd9f6f176504b48059b6b8b"; }
// OBFUSCATED: ::llvm::cl::desc(R"PO()PO")
// OBFUSCATED: getArgument() const override { return "_ba8fce64443d8c815b0ed7477793587c"; }
// OBFUSCATED: getDescription() const override { return R"PD()PD"; }
// OBFUSCATED: getName() const override { return "_6b8d928968d2a98e9bc5af1bffb90f20"; }
// OBFUSCATED-NOT: First pass summary
// OBFUSCATED-NOT: First option description.
// OBFUSCATED-NOT: Second pass summary
// OBFUSCATED-NOT: return "first-test-pass"
// OBFUSCATED-NOT: return "FirstTestPass"
// OBFUSCATED-NOT: return "second-test-pass"
// OBFUSCATED-NOT: return "SecondTestPass"
// OBFUSCATED: inline void registerFirstTestPass()
// OBFUSCATED: inline void registerSecondTestPass()

// C-API entry points are emitted for every pass. The C symbol uses the
// pass's C++ class name (unaffected by obfuscation) so downstream linkage
// is stable across builds with different obfuscators.
// CAPI: mlirCreateTestFirstTestPass
// CAPI: mlirRegisterTestFirstTestPass
// CAPI: mlirCreateTestSecondTestPass
// CAPI: mlirRegisterTestSecondTestPass
