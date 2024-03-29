//===- IntrinsicsDirectX.td - Defines DirectX intrinsics ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the DirectX-specific intrinsics.
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "hlsl" in {
    def int_hlsl_all  : DefaultAttrsIntrinsic<[llvm_i1_ty], [llvm_any_ty]>;

    def int_hlsl_thread_id : Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem, IntrWillReturn]>;
}
