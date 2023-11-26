; Check that MC/DC intrinsics are properly lowered
; RUN: opt < %s -passes=instrprof -S | FileCheck %s
; RUN: opt < %s -passes=instrprof -runtime-counter-relocation -S 2>&1 | FileCheck %s --check-prefix RELOC

; RELOC: Runtime counter relocation is presently not supported for MC/DC bitmaps

target triple = "x86_64-unknown-linux-gnu"

@__profn_test = private constant [4 x i8] c"test"

; CHECK: @__profbm_test = private global [1 x i8] zeroinitializer, section "__llvm_prf_bits", comdat, align 1

define dso_local void @test(i32 noundef %A) {
entry:
  %A.addr = alloca i32, align 4
  %mcdc.addr = alloca i32, align 4
  call void @llvm.instrprof.cover(ptr @__profn_test, i64 99278, i32 5, i32 0)
  ; CHECK: store i8 0, ptr @__profc_test, align 1

  call void @llvm.instrprof.mcdc.parameters(ptr @__profn_test, i64 99278, i32 1)
  store i32 0, ptr %mcdc.addr, align 4
  %0 = load i32, ptr %A.addr, align 4
  %tobool = icmp ne i32 %0, 0

  call void @llvm.instrprof.mcdc.condbitmap.update(ptr @__profn_test, i64 99278, i32 0, ptr %mcdc.addr, i1 %tobool)
  ; CHECK:      %mcdc.temp = load i32, ptr %mcdc.addr, align 4
  ; CHECK-NEXT: %1 = zext i1 %tobool to i32
  ; CHECK-NEXT: %2 = shl i32 %1, 0
  ; CHECK-NEXT: %3 = or i32 %mcdc.temp, %2
  ; CHECK-NEXT: store i32 %3, ptr %mcdc.addr, align 4

  call void @llvm.instrprof.mcdc.tvbitmap.update(ptr @__profn_test, i64 99278, i32 1, i32 0, ptr %mcdc.addr)
  ; CHECK:       %mcdc.temp1 = load i32, ptr %mcdc.addr, align 4
  ; CHECK-NEXT: %4 = lshr i32 %mcdc.temp1, 3
  ; CHECK-NEXT: %5 = zext i32 %4 to i64
  ; CHECK-NEXT: %6 = add i64 ptrtoint (ptr @__profbm_test to i64), %5
  ; CHECK-NEXT: %7 = inttoptr i64 %6 to ptr
  ; CHECK-NEXT: %8 = and i32 %mcdc.temp1, 7
  ; CHECK-NEXT: %9 = trunc i32 %8 to i8
  ; CHECK-NEXT: %10 = shl i8 1, %9
  ; CHECK-NEXT: %mcdc.bits = load i8, ptr %7, align 1
  ; CHECK-NEXT: %11 = or i8 %mcdc.bits, %10
  ; CHECK-NEXT: store i8 %11, ptr %7, align 1
  ret void
}

declare void @llvm.instrprof.cover(ptr, i64, i32, i32)

declare void @llvm.instrprof.mcdc.parameters(ptr, i64, i32)

declare void @llvm.instrprof.mcdc.condbitmap.update(ptr, i64, i32, ptr, i1)

declare void @llvm.instrprof.mcdc.tvbitmap.update(ptr, i64, i32, i32, ptr)
