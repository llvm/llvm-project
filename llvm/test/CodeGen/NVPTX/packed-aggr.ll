; RUN: not --crash llc < %s -march=nvptx -mcpu=sm_20 -mattr=+ptx70 2>&1 | \
; RUN:   FileCheck %s --check-prefix=ERROR

; RUN: llc < %s -march=nvptx -mcpu=sm_20 -mattr=+ptx71 | \
; RUN:   FileCheck %s --check-prefixes=CHECK,CHECK32
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 -mattr=+ptx71 | \
; RUN:   FileCheck %s --check-prefixes=CHECK,CHECK64
; RUN: %if ptxas-11.1 && !ptxas-12.0%{ llc < %s -march=nvptx -mcpu=sm_20 -mattr=+ptx71 | %ptxas-verify %}
; RUN: %if ptxas-11.1 %{ llc < %s -march=nvptx64 -mcpu=sm_20 -mattr=+ptx71 | %ptxas-verify %}

;; Test that packed structs with symbol references are represented using the
;; mask() operator.

declare void @func()
@p = addrspace(1) global i8 0
; CHECK: .extern .func func
; CHECK: .u8 p;

%t1 = type <{ i16, ptr, i8, ptr, ptr, i32 }>
@s1 = addrspace(1) global %t1 <{
; ERROR: initialized packed aggregate with pointers 's1' requires at least PTX ISA version 7.1
; CHECK32: .global .align 1 .u8 s1[19] = {
; CHECK64: .global .align 1 .u8 s1[31] = {
    i16 12,
; CHECK-SAME:   12, 0,
    ptr addrspacecast (ptr addrspace(1) @p to ptr),
; CHECK-SAME:   0xFF(generic(p)), 0xFF00(generic(p)), 0xFF0000(generic(p)), 0xFF000000(generic(p)),
; CHECK64-SAME: 0xFF00000000(generic(p)), 0xFF0000000000(generic(p)), 0xFF000000000000(generic(p)), 0xFF00000000000000(generic(p)),
    i8 34,
; CHECK-SAME:   34
    ptr @func,
; CHECK-SAME:   0xFF(func), 0xFF00(func), 0xFF0000(func), 0xFF000000(func),
; CHECK64-SAME: 0xFF00000000(func), 0xFF0000000000(func), 0xFF000000000000(func), 0xFF00000000000000(func),
    ptr addrspacecast (ptr addrspace(1) getelementptr (i8, ptr addrspace(1) @p, i32 3) to ptr),
; CHECK-SAME:   0xFF(generic(p)+3), 0xFF00(generic(p)+3), 0xFF0000(generic(p)+3), 0xFF000000(generic(p)+3),
; CHECK64-SAME: 0xFF00000000(generic(p)+3), 0xFF0000000000(generic(p)+3), 0xFF000000000000(generic(p)+3), 0xFF00000000000000(generic(p)+3),
    i32 56 }>, align 1
; CHECK-SAME:   56, 0, 0, 0};

;; Test a case than an unaligned pointer is in a nested struct.

%t2i = type <{ ptr }>
%t2o = type { i8, %t2i, i32 }
@s2 = addrspace(1) global %t2o {
; CHECK32: .global .align 8 .u8 s2[12] = {
; CHECK64: .global .align 8 .u8 s2[16] = {
    i8 12,
; CHECK-SAME:   12,
    %t2i <{ ptr @func }>,
; CHECK-SAME:   0xFF(func), 0xFF00(func), 0xFF0000(func), 0xFF000000(func),
; CHECK64-SAME: 0xFF00000000(func), 0xFF0000000000(func), 0xFF000000000000(func), 0xFF00000000000000(func),
    i32 34}
; CHECK-SAME:   0, 0, 0,
; CHECK-SAME:   34, 0, 0, 0};

;; Test that a packed struct which size is not multiple of the pointer size
;; is printed in bytes and uses the mask() operator for pointers even though
;; the pointers are aligned.

%t3 = type <{ ptr, i8 }>
@s3 = addrspace(1) global %t3 <{
; CHECK32: .global .align 1 .u8 s3[5] = {
; CHECK64: .global .align 1 .u8 s3[9] = {
    ptr @func,
; CHECK-SAME:   0xFF(func), 0xFF00(func), 0xFF0000(func), 0xFF000000(func),
; CHECK64-SAME: 0xFF00000000(func), 0xFF0000000000(func), 0xFF000000000000(func), 0xFF00000000000000(func),
    i8 56 }>, align 1
; CHECK-SAME:   56};

;; Test that a packed struct with aligned pointers is printed in words.

%t4 = type <{ ptr, i64 }>
@s4 = addrspace(1) global %t4 <{
; CHECK32: .global .align 1 .u32 s4[3] = {
; CHECK64: .global .align 1 .u64 s4[2] = {
    ptr @func,
; CHECK-SAME:   func,
    i64 15}>, align 1
; CHECK32-SAME: 15, 0};
; CHECK64-SAME: 15};

;; Test that a packed struct with unaligned pointers inside an array is handled.

%t5 = type <{ ptr, i16 }>
@a5 = addrspace(1) global [2 x %t5] [%t5 <{ ptr @func, i16 5 }>, %t5 <{ ptr @func, i16 9 }> ]
; CHECK32: .global .align 8 .u8 a5[12] = {
; CHECK32-SAME: 0xFF(func), 0xFF00(func), 0xFF0000(func), 0xFF000000(func), 5, 0,
; CHECK32-SAME: 0xFF(func), 0xFF00(func), 0xFF0000(func), 0xFF000000(func), 9, 0};
; CHECK64: .global .align 8 .u8 a5[20] = {
; CHECK64-SAME: 0xFF(func), 0xFF00(func), 0xFF0000(func), 0xFF000000(func),
; CHECK64-SAME: 0xFF00000000(func), 0xFF0000000000(func), 0xFF000000000000(func), 0xFF00000000000000(func),
; CHECK64-SAME: 5, 0,
; CHECK64-SAME: 0xFF(func), 0xFF00(func), 0xFF0000(func), 0xFF000000(func),
; CHECK64-SAME: 0xFF00000000(func), 0xFF0000000000(func), 0xFF000000000000(func), 0xFF00000000000000(func),
; CHECK64-SAME: 9, 0};
