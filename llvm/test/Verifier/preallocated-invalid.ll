; RUN: not opt -S %s -verify 2>&1 | FileCheck %s

declare token @llvm.call.preallocated.setup(i32)
declare i8* @llvm.call.preallocated.arg(token, i32)

; Fake LLVM intrinsic to return a token
declare token @llvm.what()

declare void @foo0()
declare void @foo1(i32* preallocated(i32))
declare void @foo2(i32* preallocated(i32), i32*, i32* preallocated(i32))
declare i32 @blackbox()

; CHECK: llvm.call.preallocated.arg must be called with a "preallocated" call site attribute
define void @preallocated_arg_missing_preallocated_attribute() {
    %cs = call token @llvm.call.preallocated.setup(i32 1)
    %x = call i8* @llvm.call.preallocated.arg(token %cs, i32 0)
    %y = bitcast i8* %x to i32*
    call void @foo1(i32* preallocated(i32) %y) ["preallocated"(token %cs)]
    ret void
}

; CHECK: preallocated as a call site attribute can only be on llvm.call.preallocated.arg
define void @preallocated_call_site_attribute_not_on_arg() {
    call void @foo0() preallocated(i32)
    ret void
}

; CHECK: "preallocated" argument must be a token from llvm.call.preallocated.setup
define void @preallocated_bundle_token() {
    %i = call i32 @blackbox()
    call void @foo0() ["preallocated"(i32 %i)]
    ret void
}

; CHECK: "preallocated" argument must be a token from llvm.call.preallocated.setup
define void @preallocated_bundle_token_from_setup() {
    %cs = call token @llvm.what()
    call void @foo0() ["preallocated"(token %cs)]
    ret void
}

; CHECK: Expected exactly one preallocated bundle operand
define void @preallocated_bundle_one_token() {
    %cs0 = call token @llvm.call.preallocated.setup(i32 0)
    %cs1 = call token @llvm.call.preallocated.setup(i32 0)
    call void @foo0() ["preallocated"(token %cs0, token %cs1)]
    ret void
}

; CHECK: Multiple preallocated operand bundles
define void @preallocated_multiple_bundles() {
    %cs0 = call token @llvm.call.preallocated.setup(i32 0)
    %cs1 = call token @llvm.call.preallocated.setup(i32 0)
    call void @foo0() ["preallocated"(token %cs0), "preallocated"(token %cs1)]
    ret void
}

; CHECK: Can have at most one call
define void @preallocated_one_call() {
    %cs = call token @llvm.call.preallocated.setup(i32 1)
    %x = call i8* @llvm.call.preallocated.arg(token %cs, i32 0) preallocated(i32)
    %y = bitcast i8* %x to i32*
    call void @foo1(i32* preallocated(i32) %y) ["preallocated"(token %cs)]
    call void @foo1(i32* preallocated(i32) %y) ["preallocated"(token %cs)]
    ret void
}

; CHECK: must be a constant
define void @preallocated_setup_constant() {
    %ac = call i32 @blackbox()
    %cs = call token @llvm.call.preallocated.setup(i32 %ac)
    ret void
}

; CHECK: must be between 0 and corresponding
define void @preallocated_setup_arg_index_in_bounds() {
    %cs = call token @llvm.call.preallocated.setup(i32 2)
    %a0 = call i8* @llvm.call.preallocated.arg(token %cs, i32 2) preallocated(i32)
    ret void
}

; CHECK: Attribute 'preallocated' type does not match parameter
define void @preallocated_attribute_type_mismatch() {
    %cs = call token @llvm.call.preallocated.setup(i32 1)
    %x = call i8* @llvm.call.preallocated.arg(token %cs, i32 0) preallocated(i32)
    %y = bitcast i8* %x to i32*
    call void @foo1(i32* preallocated(i8) %y) ["preallocated"(token %cs)]
    ret void
}

; CHECK: preallocated operand requires a preallocated bundle
define void @preallocated_require_bundle() {
    %cs = call token @llvm.call.preallocated.setup(i32 1)
    %x = call i8* @llvm.call.preallocated.arg(token %cs, i32 0) preallocated(i32)
    %y = bitcast i8* %x to i32*
    call void @foo1(i32* preallocated(i32) %y)
    ret void
}

; CHECK: arg size must be equal to number of preallocated arguments
define void @preallocated_num_args() {
    %cs = call token @llvm.call.preallocated.setup(i32 3)
    %x = call i8* @llvm.call.preallocated.arg(token %cs, i32 0) preallocated(i32)
    %x1 = bitcast i8* %x to i32*
    %y = call i8* @llvm.call.preallocated.arg(token %cs, i32 1) preallocated(i32)
    %y1 = bitcast i8* %y to i32*
    %a = inttoptr i32 0 to i32*
    call void @foo2(i32* preallocated(i32) %x1, i32* %a, i32* preallocated(i32) %y1) ["preallocated"(token %cs)]
    ret void
}

; CHECK: token argument must be a llvm.call.preallocated.setup
define void @preallocated_arg_token() {
    %t = call token @llvm.what()
    %x = call i8* @llvm.call.preallocated.arg(token %t, i32 1) preallocated(i32)
    ret void
}

; CHECK: musttail and preallocated not yet supported
define void @musttail() {
    %cs = call token @llvm.call.preallocated.setup(i32 0)
    musttail call void @foo0() ["preallocated"(token %cs)]
    ret void
}
