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

  call void @llvm.instrprof.mcdc.tvbitmap.update(ptr @__profn_test, i64 99278, i32 0, ptr %mcdc.addr)
  ; CHECK:      %[[TEMP0:mcdc.*]] = load i32, ptr %mcdc.addr, align 4
  ; CHECK-NEXT: %[[TEMP:[0-9]+]] = add i32 %[[TEMP0]], 0
  ; CHECK-NEXT: %[[LAB4:[0-9]+]] = lshr i32 %[[TEMP]], 3
  ; CHECK-NEXT: %[[LAB7:[0-9]+]] = getelementptr inbounds i8, ptr @__profbm_test, i32 %[[LAB4]]
  ; CHECK-NEXT: %[[LAB8:[0-9]+]] = and i32 %[[TEMP]], 7
  ; CHECK-NEXT: %[[LAB9:[0-9]+]] = trunc i32 %[[LAB8]] to i8
  ; CHECK-NEXT: %[[LAB10:[0-9]+]] = shl i8 1, %[[LAB9]]
  ; CHECK-NEXT: %[[BITS:mcdc.*]] = load i8, ptr %[[LAB7]], align 1
  ; CHECK-NEXT: %[[LAB11:[0-9]+]] = or i8 %[[BITS]], %[[LAB10]]
  ; CHECK-NEXT: store i8 %[[LAB11]], ptr %[[LAB7]], align 1
  ret void
}

declare void @llvm.instrprof.cover(ptr, i64, i32, i32)

declare void @llvm.instrprof.mcdc.parameters(ptr, i64, i32)

declare void @llvm.instrprof.mcdc.tvbitmap.update(ptr, i64, i32, ptr)
