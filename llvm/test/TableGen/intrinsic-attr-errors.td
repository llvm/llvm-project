// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include -DFN_ATTR %s 2>&1 | FileCheck --check-prefix=FN-ATTR %s
// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include -DPARAM_ATTR %s 2>&1 | FileCheck --check-prefix=PARAM-ATTR %s
// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include -DRET_ATTR %s 2>&1 | FileCheck --check-prefix=RET-ATTR %s

#define TEST_INTRINSICS_SUPPRESS_DEFS
include "llvm/IR/Intrinsics.td"

#ifdef FN_ATTR
// FN-ATTR: error: NoAlias is not a function attribute
def IntrBadFnAttr : IntrinsicFnAttr<NoAlias>;
def int_bad_fn_attr : Intrinsic<[], [], [IntrBadFnAttr]>;
#endif

#ifdef PARAM_ATTR
// PARAM-ATTR: error: Cold is not a ParamAttr
def int_bad_param_attr : Intrinsic<[], [llvm_i32_ty],
                                   [IntrinsicArgAttr<ArgIndex<0>, Cold>]>;
#endif

#ifdef RET_ATTR
// RET-ATTR: error: ReadOnly is not a RetAttr
def int_bad_ret_attr : Intrinsic<[llvm_ptr_ty], [],
                                 [IntrinsicArgAttr<RetIndex, ReadOnly>]>;
#endif
