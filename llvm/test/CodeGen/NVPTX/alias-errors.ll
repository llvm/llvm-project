; RUN: not --crash llc < %s -march=nvptx64 -mcpu=sm_30 -mattr=+ptx43 2>&1 | FileCheck %s --check-prefix=ATTR
; RUN: not --crash llc < %s -march=nvptx64 -mcpu=sm_20 -mattr=+ptx63 2>&1 | FileCheck %s --check-prefix=ATTR
; RUN: not --crash llc < %s -march=nvptx64 -mcpu=sm_30 -mattr=+ptx63 2>&1 | FileCheck %s --check-prefix=ALIAS

; ATTR: .alias requires PTX version >= 6.3 and sm_30

; ALIAS: NVPTX aliasee must be a non-kernel function
@a = global i32 42, align 8
@b = internal alias i32, ptr @a
