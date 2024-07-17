; Check that MC/DC intrinsics are properly lowered
; RUN: opt < %s -passes=instrprof -S | FileCheck %s --check-prefixes=CHECK,BASIC
; RUN: opt < %s -passes=instrprof -S -instrprof-atomic-counter-update-all | FileCheck %s --check-prefixes=CHECK,ATOMIC
; RUN: opt < %s -passes=instrprof -runtime-counter-relocation -S 2>&1 | FileCheck %s --check-prefix RELOC

; RELOC: Runtime counter relocation is presently not supported for MC/DC bitmaps

target triple = "x86_64-unknown-linux-gnu"

@__profn_test = private constant [4 x i8] c"test"

; BASIC: [[PROFBM_ADDR:@__profbm_test]] = private global [1 x i8] zeroinitializer, section "__llvm_prf_bits", comdat, align 1
; ATOMIC: [[PROFBM_ADDR:@__profbm_test]] = private global [1 x i8] zeroinitializer, section "__llvm_prf_bits", comdat, align 1

define dso_local void @test(i32 noundef %A) {
entry:
  %A.addr = alloca i32, align 4
  %mcdc.addr = alloca i32, align 4
  call void @llvm.instrprof.cover(ptr @__profn_test, i64 99278, i32 5, i32 0)
  ; BASIC: store i8 0, ptr @__profc_test, align 1

  call void @llvm.instrprof.mcdc.parameters(ptr @__profn_test, i64 99278, i32 1)
  store i32 0, ptr %mcdc.addr, align 4
  %0 = load i32, ptr %A.addr, align 4
  %tobool = icmp ne i32 %0, 0

  call void @llvm.instrprof.mcdc.tvbitmap.update(ptr @__profn_test, i64 99278, i32 0, ptr %mcdc.addr)
  ; CHECK:      %[[TEMP0:mcdc.*]] = load i32, ptr %mcdc.addr, align 4
  ; CHECK-NEXT: %[[TEMP:[0-9]+]] = add i32 %[[TEMP0]], 0
  ; CHECK-NEXT: %[[LAB4:[0-9]+]] = lshr i32 %[[TEMP]], 3
  ; CHECK-NEXT: %[[LAB7:[0-9]+]] = getelementptr inbounds i8, ptr [[PROFBM_ADDR]], i32 %[[LAB4]]
  ; CHECK-NEXT: %[[LAB8:[0-9]+]] = and i32 %[[TEMP]], 7
  ; CHECK-NEXT: %[[LAB9:[0-9]+]] = trunc i32 %[[LAB8]] to i8
  ; CHECK-NEXT: %[[LAB10:[0-9]+]] = shl i8 1, %[[LAB9]]
  ; CHECK-NEXT: call void @[[RMW_OR:.+]](ptr %[[LAB7]], i8 %[[LAB10]])
  ret void
}

; CHECK: define private void @[[RMW_OR]](ptr %[[ARGPTR:.+]], i8 %[[ARGVAL:.+]])
; CHECK:      %[[BITS:.+]] = load i8, ptr %[[ARGPTR]], align 1
; BASIC-NEXT: %[[LAB11:[0-9]+]] = or i8 %[[BITS]], %[[ARGVAL]]
; BASIC-NEXT: store i8 %[[LAB11]], ptr %[[ARGPTR]], align 1
; ATOMIC-NEXT: %[[MASKED:.+]] = and i8 %[[BITS]], %[[ARGVAL]]
; ATOMIC-NEXT: %[[SHOULDWRITE:.+]] = icmp ne i8 %[[MASKED]], %[[ARGVAL]]
; ATOMIC-NEXT: br i1 %[[SHOULDWRITE]], label %[[WRITE:.+]], label %[[SKIP:.+]], !prof ![[MDPROF:[0-9]+]]
; ATOMIC: [[WRITE]]:
; ATOMIC-NEXT: %{{.+}} = atomicrmw or ptr %[[ARGPTR]], i8 %[[ARGVAL]] monotonic, align 1
; ATOMIC-NEXT: ret void
; ATOMIC: [[SKIP]]:
; CHECK-NEXT: ret void

; ATOMIC: ![[MDPROF]] = !{!"branch_weights", i32 1, i32 1048575}

declare void @llvm.instrprof.cover(ptr, i64, i32, i32)

declare void @llvm.instrprof.mcdc.parameters(ptr, i64, i32)

declare void @llvm.instrprof.mcdc.tvbitmap.update(ptr, i64, i32, ptr)
