; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mcpu=mvp -mattr=-atomics | FileCheck %s
; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mcpu=mvp -mattr=+atomics | FileCheck %s

; Test that globals assemble as expected.

target triple = "wasm32-unknown-unknown"

; CHECK-NOT: llvm.used
; CHECK-NOT: llvm.metadata
@llvm.used = appending global [1 x i32*] [i32* @g], section "llvm.metadata"

; CHECK: foo:
; CHECK: i32.const $push0=, 0{{$}}
; CHECK-NEXT: i32.load $push1=, answer($pop0){{$}}
; CHECK-NEXT: return $pop1{{$}}
define i32 @foo() {
  %a = load i32, i32* @answer
  ret i32 %a
}

; CHECK-LABEL: call_memcpy:
; CHECK-NEXT: .functype call_memcpy (i32, i32, i32) -> (i32){{$}}
; CHECK-NEXT: call            $push0=, memcpy, $0, $1, $2{{$}}
; CHECK-NEXT: return          $pop0{{$}}
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture readonly, i32, i1)
define i8* @call_memcpy(i8* %p, i8* nocapture readonly %q, i32 %n) {
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* %p, i8* %q, i32 %n, i1 false)
  ret i8* %p
}

; CHECK: .type   .Lg,@object
; CHECK: .p2align  2, 0x0{{$}}
; CHECK-NEXT: .Lg:
; CHECK-NEXT: .int32 1337{{$}}
; CHECK-NEXT: .size .Lg, 4{{$}}
@g = private global i32 1337

; CHECK-LABEL: ud:
; CHECK-NEXT: .skip 4{{$}}
; CHECK-NEXT: .size ud, 4{{$}}
@ud = internal global i32 undef

; CHECK: .type nil,@object
; CHECK: .p2align 2, 0x0
; CHECK: nil:
; CHECK: .int32 0
; CHECK: .size nil, 4
@nil = internal global i32 zeroinitializer

; CHECK: .type z,@object
; CHECK: .p2align 2, 0x0
; CHECK: z:
; CHECK: .int32 0
; CHECK: .size z, 4
@z = internal global i32 0

; CHECK: .type one,@object
; CHECK: .p2align 2, 0x0{{$}}
; CHECK-NEXT: one:
; CHECK-NEXT: .int32 1{{$}}
; CHECK-NEXT: .size one, 4{{$}}
@one = internal global i32 1

; CHECK: .type answer,@object
; CHECK: .p2align 2, 0x0{{$}}
; CHECK-NEXT: answer:
; CHECK-NEXT: .int32 42{{$}}
; CHECK-NEXT: .size answer, 4{{$}}
@answer = internal global i32 42

; CHECK: .type u32max,@object
; CHECK: .p2align 2, 0x0{{$}}
; CHECK-NEXT: u32max:
; CHECK-NEXT: .int32 4294967295{{$}}
; CHECK-NEXT: .size u32max, 4{{$}}
@u32max = internal global i32 -1

; CHECK: .type ud64,@object
; CHECK: .p2align 3, 0x0{{$}}
; CHECK-NEXT: ud64:
; CHECK-NEXT: .skip 8{{$}}
; CHECK-NEXT: .size ud64, 8{{$}}
@ud64 = internal global i64 undef

; CHECK: .type nil64,@object
; CHECK: .p2align 3, 0x0{{$}}
; CHECK-NEXT: nil64:
; CHECK-NEXT: .int64 0{{$}}
; CHECK-NEXT: .size nil64, 8{{$}}
@nil64 = internal global i64 zeroinitializer

; CHECK: .type z64,@object
; CHECK: .p2align 3, 0x0{{$}}
; CHECK-NEXT: z64:
; CHECK-NEXT: .int64 0{{$}}
; CHECK-NEXT: .size z64, 8{{$}}
@z64 = internal global i64 0

; CHECK: .type twoP32,@object
; CHECK: .p2align 3, 0x0{{$}}
; CHECK-NEXT: twoP32:
; CHECK-NEXT: .int64 4294967296{{$}}
; CHECK-NEXT: .size twoP32, 8{{$}}
@twoP32 = internal global i64 4294967296

; CHECK: .type u64max,@object
; CHECK: .p2align 3, 0x0{{$}}
; CHECK-NEXT: u64max:
; CHECK-NEXT: .int64 -1{{$}}
; CHECK-NEXT: .size u64max, 8{{$}}
@u64max = internal global i64 -1

; CHECK: .type f32ud,@object
; CHECK: .p2align 2, 0x0{{$}}
; CHECK-NEXT: f32ud:
; CHECK-NEXT: .skip 4{{$}}
; CHECK-NEXT: .size f32ud, 4{{$}}
@f32ud = internal global float undef

; CHECK: .type f32nil,@object
; CHECK: .p2align 2, 0x0{{$}}
; CHECK-NEXT: f32nil:
; CHECK-NEXT: .int32 0x00000000{{$}}
; CHECK-NEXT: .size f32nil, 4{{$}}
@f32nil = internal global float zeroinitializer

; CHECK: .type f32z,@object
; CHECK: .p2align 2, 0x0{{$}}
; CHECK-NEXT: f32z:
; CHECK-NEXT: .int32 0x00000000{{$}}
; CHECK-NEXT: .size f32z, 4{{$}}
@f32z = internal global float 0.0

; CHECK: .type f32nz,@object
; CHECK: .p2align 2, 0x0{{$}}
; CHECK: f32nz:
; CHECK: .int32 0x80000000{{$}}
; CHECK: .size f32nz, 4{{$}}
@f32nz = internal global float -0.0

; CHECK: .type f32two,@object
; CHECK: .p2align 2, 0x0{{$}}
; CHECK-NEXT: f32two:
; CHECK-NEXT: .int32 0x40000000{{$}}
; CHECK-NEXT: .size f32two, 4{{$}}
@f32two = internal global float 2.0

; CHECK: .type f64ud,@object
; CHECK: .p2align 3, 0x0{{$}}
; CHECK-NEXT: f64ud:
; CHECK-NEXT: .skip 8{{$}}
; CHECK-NEXT: .size f64ud, 8{{$}}
@f64ud = internal global double undef

; CHECK: .type f64nil,@object
; CHECK: .p2align 3, 0x0{{$}}
; CHECK-NEXT: f64nil:
; CHECK-NEXT: .int64 0x0000000000000000{{$}}
; CHECK-NEXT: .size f64nil, 8{{$}}
@f64nil = internal global double zeroinitializer

; CHECK: .type f64z,@object
; CHECK: .p2align 3, 0x0{{$}}
; CHECK-NEXT: f64z:
; CHECK-NEXT: .int64 0x0000000000000000{{$}}
; CHECK-NEXT: .size f64z, 8{{$}}
@f64z = internal global double 0.0

; CHECK: .type f64nz,@object
; CHECK: .p2align 3, 0x0{{$}}
; CHECK-NEXT: f64nz:
; CHECK-NEXT: .int64 0x8000000000000000{{$}}
; CHECK-NEXT: .size f64nz, 8{{$}}
@f64nz = internal global double -0.0

; CHECK: .type f64two,@object
; CHECK: .p2align 3, 0x0{{$}}
; CHECK-NEXT: f64two:
; CHECK-NEXT: .int64 0x4000000000000000{{$}}
; CHECK-NEXT: .size f64two, 8{{$}}
@f64two = internal global double 2.0

; Indexing into a global array produces a relocation.
; CHECK:      .type arr,@object
; CHECK:      .type ptr,@object
; CHECK:      ptr:
; CHECK-NEXT: .int32 arr+80
; CHECK-NEXT: .size ptr, 4
@arr = global [128 x i32] zeroinitializer, align 16
@ptr = global i32* getelementptr inbounds ([128 x i32], [128 x i32]* @arr, i32 0, i32 20), align 4

; Constant global.
; CHECK: .type    rom,@object{{$}}
; CHECK: .section .rodata.rom,""
; CHECK: .globl   rom{{$}}
; CHECK: .p2align   4, 0x0{{$}}
; CHECK: rom:
; CHECK: .skip    512{{$}}
; CHECK: .size    rom, 512{{$}}
@rom = constant [128 x i32] zeroinitializer, align 16

; CHECK: .type       array,@object
; CHECK: array:
; CHECK-NEXT: .skip       8
; CHECK-NEXT: .size       array, 8
; CHECK: .type       pointer_to_array,@object
; CHECK-NEXT: .section    .rodata.pointer_to_array,""
; CHECK-NEXT: .globl      pointer_to_array
; CHECK-NEXT: .p2align      2, 0x0
; CHECK-NEXT: pointer_to_array:
; CHECK-NEXT: .int32      array+4
; CHECK-NEXT: .size       pointer_to_array, 4
@array = internal constant [8 x i8] zeroinitializer, align 1
@pointer_to_array = constant i8* getelementptr inbounds ([8 x i8], [8 x i8]* @array, i32 0, i32 4), align 4

; Handle external objects with opaque type.
%struct.ASTRUCT = type opaque
@g_struct = external global %struct.ASTRUCT, align 1
define i32 @address_of_opaque()  {
  ret i32 ptrtoint (%struct.ASTRUCT* @g_struct to i32)
}
