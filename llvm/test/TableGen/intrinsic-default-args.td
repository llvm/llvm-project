// Verify the default-argument values table and lookup function generated for
// intrinsics that declare DefaultValue on an ImmArg, and the TableGen-time
// validation of those defaults.

// RUN: llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST_INTRINSICS_SUPPRESS_DEFS | FileCheck %s
// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST_INTRINSICS_SUPPRESS_DEFS -DERROR_NEGATIVE 2>&1 | FileCheck %s --check-prefix=ERR-NEGATIVE
// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST_INTRINSICS_SUPPRESS_DEFS -DERROR_RANGE 2>&1 | FileCheck %s --check-prefix=ERR-RANGE
// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST_INTRINSICS_SUPPRESS_DEFS -DERROR_NONINT 2>&1 | FileCheck %s --check-prefix=ERR-NONINT
// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST_INTRINSICS_SUPPRESS_DEFS -DERROR_OVERLOADED 2>&1 | FileCheck %s --check-prefix=ERR-OVERLOADED
// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST_INTRINSICS_SUPPRESS_DEFS -DERROR_GAP 2>&1 | FileCheck %s --check-prefix=ERR-GAP

include "llvm/IR/Intrinsics.td"

// One trailing default: arg 1 (i32) defaults to 255.
def int_test_one_default :
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
              [ImmArg<ArgIndex<1>, DefaultValue<255>>]>;

// Two trailing defaults: arg 1 (i64) = 50, arg 2 (i32) = 14.
def int_test_two_defaults :
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i64_ty, llvm_i32_ty],
              [ImmArg<ArgIndex<1>, DefaultValue<50>>,
               ImmArg<ArgIndex<2>, DefaultValue<14>>]>;

// No defaults: must map to the sentinel at offset 0.
def int_test_no_default :
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty], []>;

// The values table: offset 0 is the sentinel, then the deduplicated sequences.
// Each real sequence is [Header = (NumDefaults << 32) | FirstDefault, vals...].
//   8589934593 = (2 << 32) | 1  -> two_defaults, values 50, 14
//   4294967297 = (1 << 32) | 1  -> one_default,  value 255
// CHECK:      static constexpr uint64_t DefaultArgValuesTable[] = {
// CHECK-NEXT:   0, // offset 0: sentinel for intrinsics without defaults
// CHECK-NEXT:   {{.*}}8589934593,{{.*}}50,{{.*}}14,{{.*}}0,
// CHECK-NEXT:   {{.*}}4294967297,{{.*}}255,{{.*}}0,
// CHECK-NEXT: };

// The per-intrinsic offset table (indexed by intrinsic ID, alphabetical):
//   not_intrinsic     -> 0
//   test_no_default   -> 0 (sentinel)
//   test_one_default  -> 5
//   test_two_defaults -> 1
// CHECK:      static constexpr uint32_t DefaultArgValuesTableOffset[] = {
// CHECK-NEXT:   0, // not_intrinsic
// CHECK-NEXT:   0,
// CHECK-NEXT:   5,
// CHECK-NEXT:   1,
// CHECK-NEXT: };

// The lookup function.
// CHECK:      Intrinsic::getAllDefaultArgValues(ID IID) {
// CHECK:        if (NumDefaults == 0)
// CHECK-NEXT:     return {0, {}};

// A negative default is rejected (values are the raw unsigned bit pattern).
#ifdef ERROR_NEGATIVE
// ERR-NEGATIVE: error: default argument value -1 on parameter 1 must be non-negative
def int_test_negative :
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i8_ty],
              [ImmArg<ArgIndex<1>, DefaultValue<-1>>]>;
#endif

// A default that does not fit the parameter width is rejected.
#ifdef ERROR_RANGE
// ERR-RANGE: error: default argument value 1000 out of range for i8 parameter 1
def int_test_range :
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i8_ty],
              [ImmArg<ArgIndex<1>, DefaultValue<1000>>]>;
#endif

// A default on a non-integer parameter is rejected.
#ifdef ERROR_NONINT
// ERR-NONINT: error: default argument on parameter 1 requires an integer parameter type
def int_test_nonint :
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_float_ty],
              [ImmArg<ArgIndex<1>, DefaultValue<1>>]>;
#endif

// A default on an overloaded intrinsic is rejected (not yet supported).
#ifdef ERROR_OVERLOADED
// ERR-OVERLOADED: error: default argument values are not supported for overloaded intrinsics
def int_test_overloaded :
    Intrinsic<[llvm_anyint_ty], [llvm_anyint_ty, llvm_i32_ty],
              [ImmArg<ArgIndex<1>, DefaultValue<5>>]>;
#endif

// Defaults must form a contiguous trailing block.
#ifdef ERROR_GAP
// ERR-GAP: error: missing default argument on parameter 2
def int_test_gap :
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [ImmArg<ArgIndex<1>, DefaultValue<5>>]>;
#endif
