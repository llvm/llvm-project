; RUN: llc -mtriple=arm64_32-apple-ios7.0 %s -filetype=obj -o - -disable-post-ra -frame-pointer=non-leaf | \
; RUN:     llvm-objdump --private-headers - | \
; RUN:     FileCheck %s --check-prefix=CHECK-MACHO
; RUN: llc -mtriple=arm64_32-apple-ios7.0 %s -o - -aarch64-enable-atomic-cfg-tidy=0 -disable-post-ra -frame-pointer=non-leaf | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-OPT
; RUN: llc -mtriple=arm64_32-apple-ios7.0 %s -o - -fast-isel -aarch64-enable-atomic-cfg-tidy=0 -disable-post-ra -frame-pointer=non-leaf | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-FAST

; CHECK-MACHO: Mach header
; CHECK-MACHO: MH_MAGIC ARM64_32 V8

@var64 = global i64 zeroinitializer, align 8
@var32 = global i32 zeroinitializer, align 4

@var_got = external global i8

define ptr @test_global_addr() {
; CHECK-LABEL: test_global_addr:
; CHECK: adrp [[PAGE:x[0-9]+]], _var32@PAGE
; CHECK-OPT: add x0, [[PAGE]], _var32@PAGEOFF
; CHECK-FAST: add [[TMP:x[0-9]+]], [[PAGE]], _var32@PAGEOFF
; CHECK-FAST: and x0, [[TMP]], #0xffffffff
  ret ptr @var32
}

; ADRP is necessarily 64-bit. The important point to check is that, however that
; gets truncated to 32-bits, it's free. No need to zero out higher bits of that
; register.
define i64 @test_global_addr_extension() {
; CHECK-LABEL: test_global_addr_extension:
; CHECK: adrp [[PAGE:x[0-9]+]], _var32@PAGE
; CHECK: add x0, [[PAGE]], _var32@PAGEOFF
; CHECK-NOT: and
; CHECK: ret

  ret i64 ptrtoint(ptr @var32 to i64)
}

define i32 @test_global_value() {
; CHECK-LABEL: test_global_value:
; CHECK: adrp x[[PAGE:[0-9]+]], _var32@PAGE
; CHECK: ldr w0, [x[[PAGE]], _var32@PAGEOFF]
  %val = load i32, ptr @var32, align 4
  ret i32 %val
}

; Because the addition may wrap, it is not safe to use "ldr w0, [xN, #32]" here.
define i32 @test_unsafe_indexed_add() {
; CHECK-LABEL: test_unsafe_indexed_add:
; CHECK: add x[[VAR32:[0-9]+]], {{x[0-9]+}}, _var32@PAGEOFF
; CHECK: add w[[ADDR:[0-9]+]], w[[VAR32]], #32
; CHECK: ldr w0, [x[[ADDR]]]
  %addr_int = ptrtoint ptr @var32 to i32
  %addr_plus_32 = add i32 %addr_int, 32
  %addr = inttoptr i32 %addr_plus_32 to ptr
  %val = load i32, ptr %addr, align 4
  ret i32 %val
}

; Since we've promised there is no unsigned overflow, @var32 must be at least
; 32-bytes below 2^32, and we can use the load this time.
define i32 @test_safe_indexed_add() {
; CHECK-LABEL: test_safe_indexed_add:
; CHECK: add x[[VAR32:[0-9]+]], {{x[0-9]+}}, _var32@PAGEOFF
; CHECK: add w[[ADDR:[0-9]+]], w[[VAR32]], #32
; CHECK: ldr w0, [x[[ADDR]]]
  %addr_int = ptrtoint ptr @var32 to i64
  %addr_plus_32 = add nuw i64 %addr_int, 32
  %addr = inttoptr i64 %addr_plus_32 to ptr
  %val = load i32, ptr %addr, align 4
  ret i32 %val
}

define i32 @test_safe_indexed_or(i32 %in) {
; CHECK-LABEL: test_safe_indexed_or:
; CHECK: and [[TMP:w[0-9]+]], {{w[0-9]+}}, #0xfffffff0
; CHECK: orr w[[ADDR:[0-9]+]], [[TMP]], #0x4
; CHECK: ldr w0, [x[[ADDR]]]
  %addr_int = and i32 %in, -16
  %addr_plus_4 = or i32 %addr_int, 4
  %addr = inttoptr i32 %addr_plus_4 to ptr
  %val = load i32, ptr %addr, align 4
  ret i32 %val
}


; Promising nsw is not sufficient because the addressing mode basically
; calculates "zext(base) + zext(offset)" and nsw only guarantees
; "sext(base) + sext(offset) == base + offset".
define i32 @test_unsafe_nsw_indexed_add() {
; CHECK-LABEL: test_unsafe_nsw_indexed_add:
; CHECK: add x[[VAR32:[0-9]+]], {{x[0-9]+}}, _var32@PAGEOFF
; CHECK: add w[[ADDR:[0-9]+]], w[[VAR32]], #32
; CHECK-NOT: ubfx
; CHECK: ldr w0, [x[[ADDR]]]
  %addr_int = ptrtoint ptr @var32 to i32
  %addr_plus_32 = add nsw i32 %addr_int, 32
  %addr = inttoptr i32 %addr_plus_32 to ptr
  %val = load i32, ptr %addr, align 4
  ret i32 %val
}

; Because the addition may wrap, it is not safe to use "ldr w0, [xN, #32]" here.
define i32 @test_unsafe_unscaled_add() {
; CHECK-LABEL: test_unsafe_unscaled_add:
; CHECK: add x[[VAR32:[0-9]+]], {{x[0-9]+}}, _var32@PAGEOFF
; CHECK: add w[[ADDR:[0-9]+]], w[[VAR32]], #3
; CHECK: ldr w0, [x[[ADDR]]]
  %addr_int = ptrtoint ptr @var32 to i32
  %addr_plus_3 = add i32 %addr_int, 3
  %addr = inttoptr i32 %addr_plus_3 to ptr
  %val = load i32, ptr %addr, align 1
  ret i32 %val
}

; Since we've promised there is no unsigned overflow, @var32 must be at least
; 32-bytes below 2^32, and we can use the load this time.
define i32 @test_safe_unscaled_add() {
; CHECK-LABEL: test_safe_unscaled_add:
; CHECK: add x[[VAR32:[0-9]+]], {{x[0-9]+}}, _var32@PAGEOFF
; CHECK: add w[[ADDR:[0-9]+]], w[[VAR32]], #3
; CHECK: ldr w0, [x[[ADDR]]]
  %addr_int = ptrtoint ptr @var32 to i32
  %addr_plus_3 = add nuw i32 %addr_int, 3
  %addr = inttoptr i32 %addr_plus_3 to ptr
  %val = load i32, ptr %addr, align 1
  ret i32 %val
}

; Promising nsw is not sufficient because the addressing mode basically
; calculates "zext(base) + zext(offset)" and nsw only guarantees
; "sext(base) + sext(offset) == base + offset".
define i32 @test_unsafe_nsw_unscaled_add() {
; CHECK-LABEL: test_unsafe_nsw_unscaled_add:
; CHECK: add x[[VAR32:[0-9]+]], {{x[0-9]+}}, _var32@PAGEOFF
; CHECK: add w[[ADDR:[0-9]+]], w[[VAR32]], #3
; CHECK-NOT: ubfx
; CHECK: ldr w0, [x[[ADDR]]]
  %addr_int = ptrtoint ptr @var32 to i32
  %addr_plus_3 = add nsw i32 %addr_int, 3
  %addr = inttoptr i32 %addr_plus_3 to ptr
  %val = load i32, ptr %addr, align 1
  ret i32 %val
}

; Because the addition may wrap, it is not safe to use "ldur w0, [xN, #-3]"
; here.
define i32 @test_unsafe_negative_unscaled_add() {
; CHECK-LABEL: test_unsafe_negative_unscaled_add:
; CHECK: add x[[VAR32:[0-9]+]], {{x[0-9]+}}, _var32@PAGEOFF
; CHECK: sub w[[ADDR:[0-9]+]], w[[VAR32]], #3
; CHECK: ldr w0, [x[[ADDR]]]
  %addr_int = ptrtoint ptr @var32 to i32
  %addr_minus_3 = add i32 %addr_int, -3
  %addr = inttoptr i32 %addr_minus_3 to ptr
  %val = load i32, ptr %addr, align 1
  ret i32 %val
}

define ptr @test_got_addr() {
; CHECK-LABEL: test_got_addr:
; CHECK: adrp x[[PAGE:[0-9]+]], _var_got@GOTPAGE
; CHECK-OPT: ldr w0, [x[[PAGE]], _var_got@GOTPAGEOFF]
; CHECK-FAST: ldr w[[TMP:[0-9]+]], [x[[PAGE]], _var_got@GOTPAGEOFF]
; CHECK-FAST: and x0, x[[TMP]], #0xffffffff
  ret ptr @var_got
}

define float @test_va_arg_f32(ptr %list) {
; CHECK-LABEL: test_va_arg_f32:

; CHECK: ldr w[[START:[0-9]+]], [x0]
; CHECK: add [[AFTER:w[0-9]+]], w[[START]], #8
; CHECK: str [[AFTER]], [x0]

  ; Floating point arguments get promoted to double as per C99.
; CHECK: ldr [[DBL:d[0-9]+]], [x[[START]]]
; CHECK: fcvt s0, [[DBL]]
  %res = va_arg ptr %list, float
  ret float %res
}

; Interesting point is that the slot is 4 bytes.
define i8 @test_va_arg_i8(ptr %list) {
; CHECK-LABEL: test_va_arg_i8:

; CHECK: ldr w[[START:[0-9]+]], [x0]
; CHECK: add [[AFTER:w[0-9]+]], w[[START]], #4
; CHECK: str [[AFTER]], [x0]

  ; i8 gets promoted to int (again, as per C99).
; CHECK: ldr w0, [x[[START]]]

  %res = va_arg ptr %list, i8
  ret i8 %res
}

; Interesting point is that the slot needs aligning (again, min size is 4
; bytes).
define i64 @test_va_arg_i64(ptr %list) {
; CHECK-LABEL: test_va_arg_i64:

  ; Update the list for the next user (minimum slot size is 4, but the actual
  ; argument is 8 which had better be reflected!)
; CHECK: ldr w[[UNALIGNED_START:[0-9]+]], [x0]
; CHECK: add [[ALIGN_TMP:x[0-9]+]], x[[UNALIGNED_START]], #7
; CHECK: and x[[START:[0-9]+]], [[ALIGN_TMP]], #0x1fffffff8
; CHECK: add w[[AFTER:[0-9]+]], w[[START]], #8
; CHECK: str w[[AFTER]], [x0]

; CHECK: ldr x0, [x[[START]]]

  %res = va_arg ptr %list, i64
  ret i64 %res
}

declare void @bar(...)
define void @test_va_call(i8 %l, i8 %r, float %in, ptr %ptr) {
; CHECK-LABEL: test_va_call:
; CHECK: add [[SUM:w[0-9]+]], {{w[0-9]+}}, w1

; CHECK-DAG: str w2, [sp, #32]
; CHECK-DAG: str xzr, [sp, #24]
; CHECK-DAG: str s0, [sp, #16]
; CHECK-DAG: str xzr, [sp, #8]
; CHECK-DAG: str [[SUM]], [sp]

  ; Add them to ensure real promotion occurs.
  %sum = add i8 %l, %r
  call void(...) @bar(i8 %sum, i64 0, float %in, double 0.0, ptr %ptr)
  ret void
}

declare ptr @llvm.frameaddress(i32)

define ptr @test_frameaddr() {
; CHECK-LABEL: test_frameaddr:
; CHECK-OPT: ldr x0, [x29]
; CHECK-FAST: ldr [[TMP:x[0-9]+]], [x29]
; CHECK-FAST: and x0, [[TMP]], #0xffffffff
  %val = call ptr @llvm.frameaddress(i32 1)
  ret ptr %val
}

declare ptr @llvm.returnaddress(i32)

define ptr @test_toplevel_returnaddr() {
; CHECK-LABEL: test_toplevel_returnaddr:
; CHECK-OPT: mov x0, x30
; CHECK-FAST: and x0, x30, #0xffffffff
  %val = call ptr @llvm.returnaddress(i32 0)
  ret ptr %val
}

define ptr @test_deep_returnaddr() {
; CHECK-LABEL: test_deep_returnaddr:
; CHECK: ldr x[[FRAME_REC:[0-9]+]], [x29]
; CHECK-OPT: ldr x30, [x[[FRAME_REC]], #8]
; CHECK-OPT: hint #7
; CHECK-OPT: mov x0, x30
; CHECK-FAST: ldr [[TMP:x[0-9]+]], [x[[FRAME_REC]], #8]
; CHECK-FAST: and x0, [[TMP]], #0xffffffff
  %val = call ptr @llvm.returnaddress(i32 1)
  ret ptr %val
}

define void @test_indirect_call(ptr %func) {
; CHECK-LABEL: test_indirect_call:
; CHECK: blr x0
  call void() %func()
  ret void
}

; Safe to use the unextended address here
define void @test_indirect_safe_call(ptr %weird_funcs) {
; CHECK-LABEL: test_indirect_safe_call:
; CHECK: add w[[ADDR32:[0-9]+]], w0, #4
; CHECK-OPT-NOT: ubfx
; CHECK: blr x[[ADDR32]]
  %addr = getelementptr i32, ptr %weird_funcs, i32 1
  call void() %addr()
  ret void
}

declare void @simple()
define void @test_simple_tail_call() {
; CHECK-LABEL: test_simple_tail_call:
; CHECK: b _simple
  tail call void @simple()
  ret void
}

define void @test_indirect_tail_call(ptr %func) {
; CHECK-LABEL: test_indirect_tail_call:
; CHECK: br x0
  tail call void() %func()
  ret void
}

; Safe to use the unextended address here
define void @test_indirect_safe_tail_call(ptr %weird_funcs) {
; CHECK-LABEL: test_indirect_safe_tail_call:
; CHECK: add w[[ADDR32:[0-9]+]], w0, #4
; CHECK-OPT-NOT: ubfx
; CHECK-OPT: br x[[ADDR32]]
  %addr = getelementptr i32, ptr %weird_funcs, i32 1
  tail call void() %addr()
  ret void
}

; For the "armv7k" slice, Clang will be emitting some small structs as [N x
; i32]. For ABI compatibility with arm64_32 these need to be passed in *X*
; registers (e.g. [2 x i32] would be packed into a single register).

define i32 @test_in_smallstruct_low([3 x i32] %in) {
; CHECK-LABEL: test_in_smallstruct_low:
; CHECK: mov x0, x1
  %val = extractvalue [3 x i32] %in, 2
  ret i32 %val
}

define i32 @test_in_smallstruct_high([3 x i32] %in) {
; CHECK-LABEL: test_in_smallstruct_high:
; CHECK: lsr x0, x0, #32
  %val = extractvalue [3 x i32] %in, 1
  ret i32 %val
}

; The 64-bit DarwinPCS ABI has the quirk that structs on the stack are always
; 64-bit aligned. This must not happen for arm64_32 since othwerwise va_arg will
; be incompatible with the armv7k ABI.
define i32 @test_in_smallstruct_stack([8 x i64], i32, [3 x i32] %in) {
; CHECK-LABEL: test_in_smallstruct_stack:
; CHECK: ldr w0, [sp, #4]
  %val = extractvalue [3 x i32] %in, 0
  ret i32 %val
}

define [2 x i32] @test_ret_smallstruct([3 x i32] %in) {
; CHECK-LABEL: test_ret_smallstruct:
; CHECK: mov x0, #1
; CHECK: movk x0, #2, lsl #32

  ret [2 x i32] [i32 1, i32 2]
}

declare void @smallstruct_callee([4 x i32])
define void @test_call_smallstruct() {
; CHECK-LABEL: test_call_smallstruct:
; CHECK: mov x0, #1
; CHECK: movk x0, #2, lsl #32
; CHECK: mov x1, #3
; CHECK: movk x1, #4, lsl #32
; CHECK: bl _smallstruct_callee

  call void @smallstruct_callee([4 x i32] [i32 1, i32 2, i32 3, i32 4])
  ret void
}

declare void @smallstruct_callee_stack([8 x i64], i32, [2 x i32])
define void @test_call_smallstruct_stack() {
; CHECK-LABEL: test_call_smallstruct_stack:
; CHECK: mov [[VAL:x[0-9]+]], #1
; CHECK: movk [[VAL]], #2, lsl #32
; CHECK: stur [[VAL]], [sp, #4]

  call void @smallstruct_callee_stack([8 x i64] undef, i32 undef, [2 x i32] [i32 1, i32 2])
  ret void
}

declare [3 x i32] @returns_smallstruct()
define i32 @test_use_smallstruct_low() {
; CHECK-LABEL: test_use_smallstruct_low:
; CHECK: bl _returns_smallstruct
; CHECK: mov x0, x1

  %struct = call [3 x i32] @returns_smallstruct()
  %val = extractvalue [3 x i32] %struct, 2
  ret i32 %val
}

define i32 @test_use_smallstruct_high() {
; CHECK-LABEL: test_use_smallstruct_high:
; CHECK: bl _returns_smallstruct
; CHECK: lsr x0, x0, #32

  %struct = call [3 x i32] @returns_smallstruct()
  %val = extractvalue [3 x i32] %struct, 1
  ret i32 %val
}

; If a small struct can't be allocated to x0-x7, the remaining registers should
; be marked as unavailable and subsequent GPR arguments should also be on the
; stack. Obviously the struct itself should be passed entirely on the stack.
define i32 @test_smallstruct_padding([7 x i64], [4 x i32] %struct, i32 %in) {
; CHECK-LABEL: test_smallstruct_padding:
; CHECK-DAG: ldr [[IN:w[0-9]+]], [sp, #16]
; CHECK-DAG: ldr [[LHS:w[0-9]+]], [sp]
; CHECK: add w0, [[LHS]], [[IN]]
  %lhs = extractvalue [4 x i32] %struct, 0
  %sum = add i32 %lhs, %in
  ret i32 %sum
}

declare void @take_small_smallstruct(i64, [1 x i32])
define void @test_small_smallstruct() {
; CHECK-LABEL: test_small_smallstruct:
; CHECK-DAG: mov w0, #1
; CHECK-DAG: mov w1, #2
; CHECK: bl _take_small_smallstruct
  call void @take_small_smallstruct(i64 1, [1 x i32] [i32 2])
  ret void
}

define void @test_bare_frameaddr(ptr %addr) {
; CHECK-LABEL: test_bare_frameaddr:
; CHECK: add x[[LOCAL:[0-9]+]], sp, #{{[0-9]+}}
; CHECK: str w[[LOCAL]],

  %ptr = alloca i8
  store ptr %ptr, ptr %addr, align 4
  ret void
}

define void @test_sret_use(ptr sret([8 x i64]) %out) {
; CHECK-LABEL: test_sret_use:
; CHECK: str xzr, [x8]
  store i64 0, ptr %out
  ret void
}

define i64 @test_sret_call() {
; CHECK-LABEL: test_sret_call:
; CHECK: mov x8, sp
; CHECK: bl _test_sret_use
  %arr = alloca [8 x i64]
  call void @test_sret_use(ptr sret([8 x i64]) %arr)

  %val = load i64, ptr %arr
  ret i64 %val
}

define double @test_constpool() {
; CHECK-LABEL: test_constpool:
; CHECK: adrp x[[PAGE:[0-9]+]], [[POOL:lCPI[0-9]+_[0-9]+]]@PAGE
; CHECK: ldr d0, [x[[PAGE]], [[POOL]]@PAGEOFF]
  ret double 1.0e-6
}

define ptr @test_blockaddress() {
; CHECK-LABEL: test_blockaddress:
; CHECK: [[BLOCK:Ltmp[0-9]+]]:
; CHECK: adrp x[[PAGE:[0-9]+]], lCPI{{[0-9]+_[0-9]+}}@PAGE
; CHECK: ldr x0, [x[[PAGE]], lCPI{{[0-9]+_[0-9]+}}@PAGEOFF]
  br label %dest
dest:
  ret ptr blockaddress(@test_blockaddress, %dest)
}

define ptr @test_indirectbr(ptr %dest) {
; CHECK-LABEL: test_indirectbr:
; CHECK: br x0
  indirectbr ptr %dest, [label %true, label %false]

true:
  ret ptr blockaddress(@test_indirectbr, %true)
false:
  ret ptr blockaddress(@test_indirectbr, %false)
}

; ISelDAGToDAG tries to fold an offset FI load (in this case var+4) into the
; actual load instruction. This needs to be done slightly carefully since we
; claim the FI in the process -- it doesn't need extending.
define float @test_frameindex_offset_load() {
; CHECK-LABEL: test_frameindex_offset_load:
; CHECK: ldr s0, [sp, #4]
  %arr = alloca float, i32 4, align 8
  %addr = getelementptr inbounds float, ptr %arr, i32 1

  %val = load float, ptr %addr, align 4
  ret float %val
}

define void @test_unaligned_frameindex_offset_store() {
; CHECK-LABEL: test_unaligned_frameindex_offset_store:
; CHECK: mov x[[TMP:[0-9]+]], sp
; CHECK: orr w[[ADDR:[0-9]+]], w[[TMP]], #0x2
; CHECK: mov [[VAL:w[0-9]+]], #42
; CHECK: str [[VAL]], [x[[ADDR]]]
  %arr = alloca [4 x i32]

  %addr.int = ptrtoint ptr %arr to i32
  %addr.nextint = add nuw i32 %addr.int, 2
  %addr.next = inttoptr i32 %addr.nextint to ptr
  store i32 42, ptr %addr.next
  ret void
}


define {i64, ptr} @test_pre_idx(ptr %addr) {
; CHECK-LABEL: test_pre_idx:

; CHECK: add w[[ADDR:[0-9]+]], w0, #8
; CHECK: ldr x0, [x[[ADDR]]]
  %addr.int = ptrtoint ptr %addr to i32
  %addr.next.int = add nuw i32 %addr.int, 8
  %addr.next = inttoptr i32 %addr.next.int to ptr
  %val = load i64, ptr %addr.next

  %tmp = insertvalue {i64, ptr} undef, i64 %val, 0
  %res = insertvalue {i64, ptr} %tmp, ptr %addr.next, 1

  ret {i64, ptr} %res
}

; Forming a post-indexed load is invalid here since the GEP needs to work when
; %addr wraps round to 0.
define {i64, ptr} @test_invalid_pre_idx(ptr %addr) {
; CHECK-LABEL: test_invalid_pre_idx:
; CHECK: add w1, w0, #8
; CHECK: ldr x0, [x1]
  %addr.next = getelementptr i64, ptr %addr, i32 1
  %val = load i64, ptr %addr.next

  %tmp = insertvalue {i64, ptr} undef, i64 %val, 0
  %res = insertvalue {i64, ptr} %tmp, ptr %addr.next, 1

  ret {i64, ptr} %res
}

declare void @callee(ptr)
define void @test_stack_guard() ssp {
; CHECK-LABEL: test_stack_guard:
; CHECK: adrp x[[GUARD_GOTPAGE:[0-9]+]], ___stack_chk_guard@GOTPAGE
; CHECK: ldr w[[GUARD_ADDR:[0-9]+]], [x[[GUARD_GOTPAGE]], ___stack_chk_guard@GOTPAGEOFF]
; CHECK: ldr [[GUARD_VAL:w[0-9]+]], [x[[GUARD_ADDR]]]
; CHECK: stur [[GUARD_VAL]], [x29, #[[GUARD_OFFSET:-[0-9]+]]]

; CHECK: add x0, sp, #{{[0-9]+}}
; CHECK: bl _callee

; CHECK-OPT: adrp x[[GUARD_GOTPAGE:[0-9]+]], ___stack_chk_guard@GOTPAGE
; CHECK-OPT: ldr w[[GUARD_ADDR:[0-9]+]], [x[[GUARD_GOTPAGE]], ___stack_chk_guard@GOTPAGEOFF]
; CHECK-OPT: ldr [[GUARD_VAL:w[0-9]+]], [x[[GUARD_ADDR]]]
; CHECK-OPT: ldur [[NEW_VAL:w[0-9]+]], [x29, #[[GUARD_OFFSET]]]
; CHECK-OPT: cmp [[GUARD_VAL]], [[NEW_VAL]]
; CHECK-OPT: b.ne [[FAIL:LBB[0-9]+_[0-9]+]]

; CHECK-OPT: [[FAIL]]:
; CHECK-OPT-NEXT: bl ___stack_chk_fail
  %arr = alloca [8 x i32]
  call void @callee(ptr %arr)
  ret void
}

declare i32 @__gxx_personality_v0(...)
declare void @eat_landingpad_args(i32, ptr, i32)
@_ZTI8Whatever = external global i8
define void @test_landingpad_marshalling() personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: test_landingpad_marshalling:
; CHECK-OPT: mov x2, x1
; CHECK-OPT: mov x1, x0
; CHECK: bl _eat_landingpad_args
  invoke void @callee(ptr undef) to label %done unwind label %lpad

lpad:                                             ; preds = %entry
  %exc = landingpad { ptr, i32 }
          catch ptr @_ZTI8Whatever
  %pointer = extractvalue { ptr, i32 } %exc, 0
  %selector = extractvalue { ptr, i32 } %exc, 1
  call void @eat_landingpad_args(i32 undef, ptr %pointer, i32 %selector)
  ret void

done:
  ret void
}

define void @test_dynamic_stackalloc() {
; CHECK-LABEL: test_dynamic_stackalloc:
; CHECK: sub [[REG:x[0-9]+]], sp, #32
; CHECK: mov sp, [[REG]]
; CHECK-OPT-NOT: ubfx
; CHECK: bl _callee
  br label %next

next:
  %val = alloca [8 x i32]
  call void @callee(ptr %val)
  ret void
}

define void @test_asm_memory(ptr %base.addr) {
; CHECK-LABEL: test_asm_memory:
; CHECK: add w[[ADDR:[0-9]+]], w0, #4
; CHECK: str wzr, [x[[ADDR]]
  %addr = getelementptr i32, ptr %base.addr, i32 1
  call void asm sideeffect "str wzr, $0", "*m"(ptr elementtype(i32) %addr)
  ret void
}

define void @test_unsafe_asm_memory(i64 %val) {
; CHECK-LABEL: test_unsafe_asm_memory:
; CHECK: and x[[ADDR:[0-9]+]], x0, #0xffffffff
; CHECK: str wzr, [x[[ADDR]]]
  %addr_int = trunc i64 %val to i32
  %addr = inttoptr i32 %addr_int to ptr
  call void asm sideeffect "str wzr, $0", "*m"(ptr elementtype(i32) %addr)
  ret void
}

define [9 x ptr] @test_demoted_return(ptr %in) {
; CHECK-LABEL: test_demoted_return:
; CHECK: str w0, [x8, #32]
  %res = insertvalue [9 x ptr] undef, ptr %in, 8
  ret [9 x ptr] %res
}

define ptr @test_inttoptr(i64 %in) {
; CHECK-LABEL: test_inttoptr:
; CHECK: and x0, x0, #0xffffffff
  %res = inttoptr i64 %in to ptr
  ret ptr %res
}

declare i32 @llvm.get.dynamic.area.offset.i32()
define i32 @test_dynamic_area() {
; CHECK-LABEL: test_dynamic_area:
; CHECK: mov w0, wzr
  %res = call i32 @llvm.get.dynamic.area.offset.i32()
  ret i32 %res
}

define void @test_pointer_vec_store(ptr %addr) {
; CHECK-LABEL: test_pointer_vec_store:
; CHECK: str xzr, [x0]
; CHECK-NOT: str
; CHECK-NOT: stp

  store <2 x ptr> zeroinitializer, ptr %addr, align 16
  ret void
}

define <2 x ptr> @test_pointer_vec_load(ptr %addr) {
; CHECK-LABEL: test_pointer_vec_load:
; CHECK: ldr d[[TMP:[0-9]+]], [x0]
; CHECK: ushll.2d v0, v[[TMP]], #0
  %val = load <2 x ptr>, ptr %addr, align 16
  ret <2 x ptr> %val
}

define void @test_inline_asm_mem_pointer(ptr %in) {
; CHECK-LABEL: test_inline_asm_mem_pointer:
; CHECK: str w0,
  tail call void asm sideeffect "ldr x0, $0", "m"(ptr %in)
  ret void
}


define void @test_struct_hi(i32 %hi) nounwind {
; CHECK-LABEL: test_struct_hi:
; CHECK: mov w[[IN:[0-9]+]], w0
; CHECK: bl _get_int
; CHECK-FAST-NEXT: mov w[[DST:[0-9]+]], w0
; CHECK-FAST-NEXT: orr x0, x[[DST]], x[[IN]], lsl #32
; CHECK-OPT-NEXT: bfi x0, x[[IN]], #32, #32
; CHECK-NEXT: bl _take_pair
  %val.64 = call i64 @get_int()
  %val.32 = trunc i64 %val.64 to i32

  %pair.0 = insertvalue [2 x i32] undef, i32 %val.32, 0
  %pair.1 = insertvalue [2 x i32] %pair.0, i32 %hi, 1
  call void @take_pair([2 x i32] %pair.1)

  ret void
}
declare void @take_pair([2 x i32])
declare i64 @get_int()

define i1 @test_icmp_ptr(ptr %in) {
; CHECK-LABEL: test_icmp_ptr
; CHECK: ubfx x0, x0, #31, #1
  %res = icmp slt ptr %in, null
  ret i1 %res
}

define void @test_multiple_icmp_ptr(ptr %l, ptr %r) {
; CHECK-LABEL: test_multiple_icmp_ptr:
; CHECK: tbnz w0, #31, [[FALSEBB:LBB[0-9]+_[0-9]+]]
; CHECK: tbnz w1, #31, [[FALSEBB]]
  %tst1 = icmp sgt ptr %l, inttoptr (i32 -1 to ptr)
  %tst2 = icmp sgt ptr %r, inttoptr (i32 -1 to ptr)
  %tst = and i1 %tst1, %tst2
  br i1 %tst, label %true, label %false

true:
  call void(...) @bar()
  ret void

false:
  ret void
}

define void @test_multiple_icmp_ptr_select(ptr %l, ptr %r) {
; CHECK-LABEL: test_multiple_icmp_ptr_select:
; CHECK: tbnz w0, #31, [[FALSEBB:LBB[0-9]+_[0-9]+]]
; CHECK: tbnz w1, #31, [[FALSEBB]]
  %tst1 = icmp sgt ptr %l, inttoptr (i32 -1 to ptr)
  %tst2 = icmp sgt ptr %r, inttoptr (i32 -1 to ptr)
  %tst = select i1 %tst1, i1 %tst2, i1 false
  br i1 %tst, label %true, label %false

true:
  call void(...) @bar()
  ret void

false:
  ret void
}

define ptr @test_gep_nonpow2(ptr %a0, i32 %a1) {
; CHECK-LABEL: test_gep_nonpow2:
; CHECK-OPT:      mov w[[SIZE:[0-9]+]], #18
; CHECK-OPT-NEXT: smaddl x0, w1, w[[SIZE]], x0
; CHECK-OPT-NEXT: ret

; CHECK-FAST:      mov w[[SIZE:[0-9]+]], #18
; CHECK-FAST-NEXT: smaddl [[TMP:x[0-9]+]], w1, w[[SIZE]], x0
; CHECK-FAST-NEXT: and x0, [[TMP]], #0xffffffff
; CHECK-FAST-NEXT: ret
  %tmp0 = getelementptr inbounds { [18 x i8] }, ptr %a0, i32 %a1
  ret ptr %tmp0
}

define void @test_memset(i64 %in, i8 %value)  {
; CHECK-LABEL: test_memset:
; CHECK-DAG: lsr x2, x0, #32
; CHECK-DAG: and x0, x0, #0xffffffff
; CHECK: b _memset

  %ptr.i32 = trunc i64 %in to i32
  %size.64 = lshr i64 %in, 32
  %size = trunc i64 %size.64 to i32
  %ptr = inttoptr i32 %ptr.i32 to ptr
  tail call void @llvm.memset.p0.i32(ptr align 4 %ptr, i8 %value, i32 %size, i1 false)
  ret void
}

define void @test_bzero(i64 %in)  {
; CHECK-LABEL: test_bzero:
; CHECK-DAG: lsr x1, x0, #32
; CHECK-DAG: and x0, x0, #0xffffffff
; CHECK: b _bzero

  %ptr.i32 = trunc i64 %in to i32
  %size.64 = lshr i64 %in, 32
  %size = trunc i64 %size.64 to i32
  %ptr = inttoptr i32 %ptr.i32 to ptr
  tail call void @llvm.memset.p0.i32(ptr align 4 %ptr, i8 0, i32 %size, i1 false)
  ret void
}

declare void @llvm.memset.p0.i32(ptr nocapture writeonly, i8, i32, i1)

!llvm.module.flags = !{!0}
!0 = !{i32 7, !"PIC Level", i32 2}
