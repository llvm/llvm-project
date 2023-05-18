; RUN: opt -aa-pipeline=basic-aa -passes=lint -disable-output < %s 2>&1 | FileCheck %s
target datalayout = "e-p:64:64:64"

declare fastcc void @bar()
declare void @llvm.stackrestore(ptr)
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1) nounwind
declare void @llvm.memcpy.inline.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1) nounwind
declare void @llvm.memset.p0.i8.i64(ptr nocapture, i8, i64, i1) nounwind
declare void @llvm.memset.inline.p0.i8.i64(ptr nocapture, i8, i64, i1) nounwind
declare void @has_sret(ptr sret(i8) %p)
declare void @has_noaliases(ptr noalias %p, ptr %q)
declare void @one_arg(i32)

@CG = constant i32 7
@CG2 = constant i32 7
@E = external global i8

define i32 @foo() noreturn {
  %buf = alloca i8
  %buf2 = alloca {i8, i8}, align 2
; CHECK: Caller and callee calling convention differ
  call void @bar()
; CHECK: Null pointer dereference
  store i32 0, ptr null
; CHECK: Null pointer dereference
  %t = load i32, ptr null
; CHECK: Undef pointer dereference
  store i32 0, ptr undef
; CHECK: Undef pointer dereference
  %u = load i32, ptr undef
; CHECK: All-ones pointer dereference
  store i32 0, ptr inttoptr (i64 -1 to ptr)
; CHECK: Address one pointer dereference
  store i32 0, ptr inttoptr (i64 1 to ptr)
; CHECK: Memory reference address is misaligned
  store i8 0, ptr %buf, align 2
; CHECK: Memory reference address is misaligned
  %gep = getelementptr {i8, i8}, ptr %buf2, i32 0, i32 1
  store i8 0, ptr %gep, align 2
; CHECK: Division by zero
  %sd = sdiv i32 2, 0
; CHECK: Division by zero
  %ud = udiv i32 2, 0
; CHECK: Division by zero
  %sr = srem i32 2, 0
; CHECK: Division by zero
  %ur = urem i32 2, 0
; CHECK: extractelement index out of range
  %ee = extractelement <4 x i32> zeroinitializer, i32 4
; CHECK: insertelement index out of range
  %ie = insertelement <4 x i32> zeroinitializer, i32 0, i32 4
; CHECK: Shift count out of range
  %r = lshr i32 0, 32
; CHECK: Shift count out of range
  %q = ashr i32 0, 32
; CHECK: Shift count out of range
  %l = shl i32 0, 32
; CHECK: xor(undef, undef)
  %xx = xor i32 undef, undef
; CHECK: sub(undef, undef)
  %xs = sub i32 undef, undef

; CHECK: Write to read-only memory
  store i32 8, ptr @CG
; CHECK: Write to text section
  store i32 8, ptr @foo
; CHECK: Load from block address
  %lb = load i32, ptr blockaddress(@foo, %next)
; CHECK: Call to block address
  call void() blockaddress(@foo, %next)()
; CHECK: Undefined behavior: Null pointer dereference
  call void @llvm.stackrestore(ptr null)
; CHECK: Undefined behavior: Null pointer dereference
  call void @has_sret(ptr sret(i8) null)
; CHECK: Unusual: noalias argument aliases another argument
  call void @has_noaliases(ptr @CG, ptr @CG)
; CHECK: Call argument count mismatches callee argument count
  call void (i32, i32) @one_arg(i32 0, i32 0)
; CHECK: Call argument count mismatches callee argument count
  call void () @one_arg()
; CHECK: Call argument type mismatches callee parameter type
  call void (float) @one_arg(float 0.0)

; CHECK: Write to read-only memory
call void @llvm.memcpy.p0.p0.i64(ptr @CG, ptr @CG2, i64 1, i1 0)
; CHECK: Write to read-only memory
call void @llvm.memcpy.inline.p0.p0.i64(ptr @CG, ptr @CG2, i64 1, i1 0)
; CHECK: Unusual: noalias argument aliases another argument
call void @llvm.memcpy.p0.p0.i64(ptr @CG, ptr @CG, i64 1, i1 0)

; CHECK: Write to read-only memory
call void @llvm.memset.p0.i8.i64(ptr @CG, i8 1, i64 1, i1 0)
; CHECK: Write to read-only memory
call void @llvm.memset.inline.p0.i8.i64(ptr @CG, i8 1, i64 1, i1 0)

; CHECK: Undefined behavior: Buffer overflow
  store i16 0, ptr %buf
; CHECK: Undefined behavior: Buffer overflow
  %inner = getelementptr {i8, i8}, ptr %buf2, i32 0, i32 1
  store i16 0, ptr %inner
; CHECK: Undefined behavior: Buffer overflow
  %before = getelementptr i8, ptr %buf, i32 -1
  store i16 0, ptr %before

  br label %next

next:
; CHECK: Static alloca outside of entry block
  %a = alloca i32
; CHECK: Return statement in function with noreturn attribute
  ret i32 0

foo:
; CHECK-NOT: Undefined behavior: Buffer overflow
; CHECK-NOT: Memory reference address is misaligned
  store i64 0, ptr @E
  %z = add i32 0, 0
; CHECK: unreachable immediately preceded by instruction without side effects
  unreachable
}

; CHECK: Unnamed function with non-local linkage
define void @0() nounwind {
  ret void
}

; CHECK: va_start called in a non-varargs function
declare void @llvm.va_start(ptr)
define void @not_vararg(ptr %p) nounwind {
  call void @llvm.va_start(ptr %p)
  ret void
}

; CHECK: Undefined behavior: Branch to non-blockaddress
define void @use_indbr() {
  indirectbr ptr @foo, [label %block]
block:
  unreachable
}

; CHECK: Undefined behavior: Call with "tail" keyword references alloca
declare void @tailcallee(ptr)
define void @use_tail(ptr %valist) {
  %t = alloca i8
  tail call void @tailcallee(ptr %t)
  ret void
}

; CHECK: Unusual: Returning alloca value
define ptr @return_local(i32 %n, i32 %m) {
  %t = alloca i8, i32 %n
  %s = getelementptr i8, ptr %t, i32 %m
  ret ptr %s
}

; CHECK: Unusual: Returning alloca value
define ptr @return_obscured_local() {
entry:
  %retval = alloca ptr
  %x = alloca i32
  store ptr %x, ptr %retval
  br label %next
next:
  %t0 = load ptr, ptr %retval
  %t1 = insertvalue { i32, i32, ptr } zeroinitializer, ptr %t0, 2
  %t2 = extractvalue { i32, i32, ptr } %t1, 2
  br label %exit
exit:
  %t3 = phi ptr [ %t2, %next ]
  %t5 = ptrtoint ptr %t3 to i64
  %t6 = add i64 %t5, 0
  %t7 = inttoptr i64 %t6 to ptr
  ret ptr %t7
}

; CHECK: Undefined behavior: Undef pointer dereference
define ptr @self_reference() {
entry:
  unreachable
exit:
  %t3 = phi ptr [ %t4, %exit ]
  %t4 = bitcast ptr %t3 to ptr
  %x = load volatile i32, ptr %t3
  br label %exit
}

; CHECK: Call return type mismatches callee return type
%struct = type { double, double }
declare i32 @nonstruct_callee() nounwind
define void @struct_caller() nounwind {
entry:
  call %struct @foo()

  ; CHECK: Undefined behavior: indirectbr with no destinations
  indirectbr ptr null, []
}

define i32 @memcpy_inline_same_address() noreturn {
  %buf = alloca i64, align 1
  ; CHECK: Unusual: noalias argument aliases another argument
  call void @llvm.memcpy.inline.p0.p0.i64(ptr %buf, ptr %buf, i64 1, i1 false)
  unreachable
}
