; Check that MC/DC intrinsics are properly lowered
; RUN: opt < %s -passes=instrprof -S | FileCheck %s --check-prefixes=CHECK,BASIC
; RUN: opt < %s -passes=instrprof -S -instrprof-atomic-counter-update-all | FileCheck %s --check-prefixes=CHECK,ATOMIC
; RUN: opt < %s -passes=instrprof -S -runtime-counter-relocation | FileCheck %s --check-prefixes=CHECK,RELOC

target triple = "x86_64-unknown-linux-gnu"

@__profn_test = private constant [4 x i8] c"test"

; BASIC: [[PROFBM_ADDR:@__profbm_test]] = private global [1 x i8] zeroinitializer, section "__llvm_prf_bits", comdat, align 1
; ATOMIC: [[PROFBM_ADDR:@__profbm_test]] = private global [1 x i8] zeroinitializer, section "__llvm_prf_bits", comdat, align 1

define dso_local void @test(i32 noundef %A) {
entry:
  ; RELOC: %profbm_bias = load i64, ptr @__llvm_profile_bitmap_bias, align [[#]], !invariant.load !0
  ; RELOC: %profc_bias = load i64, ptr @__llvm_profile_counter_bias, align [[#]]
  %A.addr = alloca i32, align 4
  %mcdc.addr = alloca i32, align 4
  call void @llvm.instrprof.cover(ptr @__profn_test, i64 99278, i32 5, i32 0)
  ; BASIC: store i8 0, ptr @__profc_test, align 1
  ; RELOC: %[[PROFC_INTADDR:.+]] = add i64 ptrtoint (ptr @__profc_test to i64), %profc_bias
  ; RELOC: %[[PROFC_ADDR:.+]] = inttoptr i64 %[[PROFC_INTADDR]] to ptr
  ; RELOC: store i8 0, ptr %[[PROFC_ADDR]], align 1

  call void @llvm.instrprof.mcdc.parameters(ptr @__profn_test, i64 99278, i32 1)
  store i32 0, ptr %mcdc.addr, align 4
  %0 = load i32, ptr %A.addr, align 4
  %tobool = icmp ne i32 %0, 0

  call void @llvm.instrprof.mcdc.tvbitmap.update(ptr @__profn_test, i64 99278, i32 0, ptr %mcdc.addr)
  ; RELOC:      [[PROFBM_ADDR:%.+]] = getelementptr i8, ptr @__profbm_test, i64 %profbm_bias
  ; CHECK:      %[[TEMP0:mcdc.*]] = load i32, ptr %mcdc.addr, align 4
  ; CHECK-NEXT: %[[TEMP:[0-9]+]] = add i32 %[[TEMP0]], 0
  ; CHECK-NEXT: %[[LAB4:[0-9]+]] = lshr i32 %[[TEMP]], 3
  ; CHECK-NEXT: %[[LAB7:[0-9]+]] = getelementptr inbounds i8, ptr [[PROFBM_ADDR]], i32 %[[LAB4]]
  ; CHECK-NEXT: %[[LAB8:[0-9]+]] = and i32 %[[TEMP]], 7
  ; CHECK-NEXT: %[[LAB9:[0-9]+]] = trunc i32 %[[LAB8]] to i8
  ; CHECK-NEXT: %[[LAB10:[0-9]+]] = shl i8 1, %[[LAB9]]
  ; CHECK-NEXT: %[[BITS:.+]] = load i8, ptr %[[LAB7]], align 1
  ; ATOMIC-NEXT: %[[MASKED:.+]] = and i8 %[[BITS]], %[[LAB10]]
  ; ATOMIC-NEXT: %[[SHOULDWRITE:.+]] = icmp ne i8 %[[MASKED]], %[[LAB10]]
  ; ATOMIC-NEXT: br i1 %[[SHOULDWRITE]], label %[[WRITE:.+]], label %[[SKIP:.+]], !prof ![[MDPROF:[0-9]+]]
  ; ATOMIC: [[WRITE]]:
  ; BASIC-NEXT: %[[LAB11:[0-9]+]] = or i8 %[[BITS]], %[[LAB10]]
  ; RELOC-NEXT: %[[LAB11:[0-9]+]] = or i8 %[[BITS]], %[[LAB10]]
  ; BASIC-NEXT: store i8 %[[LAB11]], ptr %[[LAB7]], align 1
  ; RELOC-NEXT: store i8 %[[LAB11]], ptr %[[LAB7]], align 1
  ; ATOMIC-NEXT: %{{.+}} = atomicrmw or ptr %[[LAB7]], i8 %[[LAB10]] monotonic, align 1
  ; ATOMIC: [[SKIP]]:
  ret void
  ; CHECK-NEXT: ret void
}

; ATOMIC: ![[MDPROF]] = !{!"branch_weights", i32 1, i32 1048575}

declare void @llvm.instrprof.cover(ptr, i64, i32, i32)

declare void @llvm.instrprof.mcdc.parameters(ptr, i64, i32)

declare void @llvm.instrprof.mcdc.tvbitmap.update(ptr, i64, i32, ptr)
