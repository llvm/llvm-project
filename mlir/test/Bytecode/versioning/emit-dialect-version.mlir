// RUN: mlir-opt %s -emit-bytecode -emit-bytecode-producer=fixed -emit-bytecode-dialect-version=builtin=0 -emit-bytecode-dialect-version=test=2.0 -o %t.0
// RUN: mlir-opt %s -emit-bytecode -emit-bytecode-producer=fixed -emit-bytecode-dialect-version=builtin=1 -emit-bytecode-dialect-version=test=2.0 -o %t.1
// RUN: not cmp -s %t.0 %t.1
// RUN: mlir-opt %t.0 | FileCheck %s
// RUN: mlir-opt %t.1 | FileCheck %s
// RUN: mlir-opt %s -emit-bytecode -emit-bytecode-producer=fixed -o %t.default
// RUN: cmp -s %t.1 %t.default
// RUN: mlir-opt %s -emit-bytecode -emit-bytecode-producer=fixed -emit-bytecode-dialect-version=builtin=1 -emit-bytecode-dialect-version=test=1.12 -o %t.old-test
// RUN: not cmp -s %t.old-test %t.1
// RUN: mlir-opt %t.old-test | FileCheck %s
// RUN: mlir-opt %s -emit-bytecode -emit-bytecode-dialect-version=test=1.12 | mlir-opt | FileCheck %s
// RUN: mlir-opt %s -emit-bytecode -emit-bytecode-version=0 -emit-bytecode-dialect-version=test=1.12 | mlir-opt | FileCheck %s
// RUN: not mlir-opt %s -emit-bytecode-dialect-version=builtin=0 2>&1 | FileCheck %s --check-prefix=NO-BYTECODE
// RUN: not mlir-opt %s -emit-bytecode -emit-bytecode-dialect-version=builtin 2>&1 | FileCheck %s --check-prefix=MALFORMED
// RUN: not mlir-opt %s -emit-bytecode -emit-bytecode-dialect-version=builtin= 2>&1 | FileCheck %s --check-prefix=MALFORMED
// RUN: not mlir-opt %s -emit-bytecode -emit-bytecode-dialect-version==0 2>&1 | FileCheck %s --check-prefix=MALFORMED
// RUN: not mlir-opt %s -emit-bytecode -emit-bytecode-dialect-version=unknown=0 2>&1 | FileCheck %s --check-prefix=UNKNOWN
// RUN: not mlir-opt %s -emit-bytecode -emit-bytecode-dialect-version=arith=0 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED-DIALECT
// RUN: not mlir-opt %s -emit-bytecode -emit-bytecode-dialect-version=func=1 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED-PARSER
// RUN: not mlir-opt %s -emit-bytecode -emit-bytecode-dialect-version=builtin=-1 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED-VERSION
// RUN: not mlir-opt %s -emit-bytecode -emit-bytecode-dialect-version=builtin=2 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED-VERSION
// RUN: not mlir-opt %s -emit-bytecode -emit-bytecode-dialect-version=builtin=abc 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED-VERSION
// RUN: not mlir-opt %s -emit-bytecode -emit-bytecode-dialect-version=builtin=0 -emit-bytecode-dialect-version=builtin=1 2>&1 | FileCheck %s --check-prefix=DUPLICATE

// NO-BYTECODE: dialect bytecode version while not emitting bytecode
// MALFORMED: expected 'dialect=version' for --emit-bytecode-dialect-version
// UNKNOWN: unknown dialect 'unknown' for bytecode emission
// UNSUPPORTED-DIALECT: dialect 'arith' does not support emitting bytecode version '0'
// UNSUPPORTED-PARSER: dialect 'func' does not support emitting bytecode version '1'
// UNSUPPORTED-VERSION: dialect 'builtin' does not support emitting bytecode version
// DUPLICATE: duplicate bytecode version for dialect 'builtin'

// CHECK: "test.versionedA"() <{dims = 123 : i64, modifier = false}>
"test.versionedA"() <{dims = 123 : i64, modifier = false}> : () -> ()
// CHECK: "test.versionedC"() <{attribute = #test.attr_params<42, 24>}>
"test.versionedC"() <{attribute = #test.attr_params<42, 24>}> : () -> ()
