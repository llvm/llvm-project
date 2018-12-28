; RUN: opt -asfix %s -S -o %t.out
; FileCheck %s --input-file %t.out
; FileCheck %s --input-file %t.out -check-prefix=CHECK-MUL
; FileCheck %s --input-file %t.out -check-prefix=CHECK-ADD
; ModuleID = 'bugpoint-reduced-simplified.ll'
source_filename = "scan.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux-sycldevice"

%"struct.std::plus" = type { i8 }
%"struct.std::multiplies" = type { i8 }

$_ZZZ8par_scanIiSt4plusIiEEvRN2cl4sycl6bufferIT_Li1ESaIS5_EEERNS3_5queueEENKUlRNS3_7handlerEE_clESC_ENKUlNS3_7nd_itemILi1EEEE_clESF_ = comdat any

$_ZZZ8par_scanIlSt10multipliesIlEEvRN2cl4sycl6bufferIT_Li1ESaIS5_EEERNS3_5queueEENKUlRNS3_7handlerEE_clESC_ENKUlNS3_7nd_itemILi1EEEE_clESF_ = comdat any

$_ZNKSt4plusIiEclERKiS2_ = comdat any

$_ZNKSt10multipliesIlEclERKlS2_ = comdat any

; Function Attrs: noinline
define linkonce_odr dso_local spir_func void @_ZZZ8par_scanIiSt4plusIiEEvRN2cl4sycl6bufferIT_Li1ESaIS5_EEERNS3_5queueEENKUlRNS3_7handlerEE_clESC_ENKUlNS3_7nd_itemILi1EEEE_clESF_(i32 addrspace(3)* %par1, i32 addrspace(3)* %par2) #0 comdat align 2 {
  %1 = alloca i32 addrspace(3)*, align 8
  %2 = alloca i32 addrspace(3)*, align 8
  %3 = alloca %"struct.std::plus", align 1
  store i32 addrspace(3)* %par1, i32 addrspace(3)** %1, align 8
  store i32 addrspace(3)* %par2, i32 addrspace(3)** %2, align 8
  %4 = load i32 addrspace(3)*, i32 addrspace(3)** %1, align 8
  %5 = load i32 addrspace(3)*, i32 addrspace(3)** %2, align 8
; CHECK:  %[[CAST1:.*]] = addrspacecast i32 addrspace(3)* %{{.*}} to i32 addrspace(4)*
  %6 = addrspacecast i32 addrspace(3)* %4 to i32*
; CHECK:  %[[CAST2:.*]] = addrspacecast i32 addrspace(3)* %{{.*}} to i32 addrspace(4)*
  %7 = addrspacecast i32 addrspace(3)* %5 to i32*
; CHECK:  %{{.*}} = call spir_func i32 @new.[[PLUS:.*]](%"struct.std::plus"* %{{.*}}, i32 addrspace(4)* %[[CAST1]], i32 addrspace(4)* %[[CAST2]])
  %8 = call spir_func i32 @_ZNKSt4plusIiEclERKiS2_(%"struct.std::plus"* %3, i32* dereferenceable(4) %6, i32* dereferenceable(4) %7)
  ret void
}

; Function Attrs: noinline
define linkonce_odr dso_local spir_func void @_ZZZ8par_scanIlSt10multipliesIlEEvRN2cl4sycl6bufferIT_Li1ESaIS5_EEERNS3_5queueEENKUlRNS3_7handlerEE_clESC_ENKUlNS3_7nd_itemILi1EEEE_clESF_(i64 addrspace(3)* %par1, i64 addrspace(3)* %par2) #0 comdat align 2 {
  %1 = alloca i64 addrspace(3)*, align 8
  %2 = alloca i64 addrspace(3)*, align 8
  %3 = alloca %"struct.std::multiplies", align 1
  store i64 addrspace(3)* %par1, i64 addrspace(3)** %1, align 8
  store i64 addrspace(3)* %par2, i64 addrspace(3)** %2, align 8
  %4 = load i64 addrspace(3)*, i64 addrspace(3)** %1, align 8
  %5 = load i64 addrspace(3)*, i64 addrspace(3)** %2, align 8
; CHECK:  %[[CAST1:.*]] = addrspacecast i64 addrspace(3)* %{{.*}} to i64 addrspace(4)*
  %6 = addrspacecast i64 addrspace(3)* %4 to i64*
; CHECK:  %[[CAST2:.*]] = addrspacecast i64 addrspace(3)* %{{.*}} to i64 addrspace(4)*
  %7 = addrspacecast i64 addrspace(3)* %5 to i64*
; CHECK:  %{{.*}} = call spir_func i64 @new.[[MUL:.*]](%"struct.std::multiplies"* %{{.*}}, i64 addrspace(4)* %[[CAST1]], i64 addrspace(4)* %[[CAST2]])
  %8 = call spir_func i64 @_ZNKSt10multipliesIlEclERKlS2_(%"struct.std::multiplies"* %3, i64* dereferenceable(8) %6, i64* dereferenceable(8) %7)
  ret void
}

; CHECK-ADD: define linkonce_odr dso_local spir_func i32 @new.[[PLUS]](%"struct.std::plus"*, i32 addrspace(4)* dereferenceable(4), i32 addrspace(4)* dereferenceable(4)) #1 align 2 {
; Function Attrs: noinline nounwind optnone
define linkonce_odr dso_local spir_func i32 @_ZNKSt4plusIiEclERKiS2_(%"struct.std::plus"*, i32* dereferenceable(4), i32* dereferenceable(4)) #1 comdat align 2 {
  %4 = alloca %"struct.std::plus"*, align 8
; CHECK-ADD:  %[[ALLOC1:.*]] = alloca i32 addrspace(4)*
  %5 = alloca i32*, align 8
; CHECK-ADD:  %[[ALLOC2:.*]] = alloca i32 addrspace(4)*
  %6 = alloca i32*, align 8
  store %"struct.std::plus"* %0, %"struct.std::plus"** %4, align 8
; CHECK-ADD:  store i32 addrspace(4)* %{{.*}}, i32 addrspace(4)** %[[ALLOC1]], align 8
  store i32* %1, i32** %5, align 8
; CHECK-ADD:  store i32 addrspace(4)* %{{.*}}, i32 addrspace(4)** %[[ALLOC2]], align 8
  store i32* %2, i32** %6, align 8
  %7 = load %"struct.std::plus"*, %"struct.std::plus"** %4, align 8
; CHECK-ADD:  %[[LOAD1:.*]] = load i32 addrspace(4)*, i32 addrspace(4)** %[[ALLOC1]]
  %8 = load i32*, i32** %5, align 8
; CHECK-ADD:  %[[NEXT_LOAD1:.*]] = load i32, i32 addrspace(4)* %[[LOAD1]]
  %9 = load i32, i32* %8, align 4
; CHECK-ADD:  %[[LOAD2:.*]] = load i32 addrspace(4)*, i32 addrspace(4)** %[[ALLOC2]]
  %10 = load i32*, i32** %6, align 8
; CHECK-ADD:  %[[NEXT_LOAD2:.*]] = load i32, i32 addrspace(4)* %[[LOAD2]]
  %11 = load i32, i32* %10, align 4
; CHECK-ADD:  %[[ADD:.*]] = add nsw i32 %[[NEXT_LOAD1]], %[[NEXT_LOAD2]]
  %12 = add nsw i32 %9, %11
; CHECK-ADD:  %[[ADD2:.*]] = add nsw i32 %[[ADD]], 1
  %13 = add nsw i32 %12, 1
; CHECK-ADD:  ret i32 %[[ADD2]]
  ret i32 %13
}

; CHECK-MUL: define linkonce_odr dso_local spir_func i64 @new.[[MUL]](%"struct.std::multiplies"*, i64 addrspace(4)* dereferenceable(8), i64 addrspace(4)* dereferenceable(8)) #1 align 2 {
; Function Attrs: noinline nounwind optnone
define linkonce_odr dso_local spir_func i64 @_ZNKSt10multipliesIlEclERKlS2_(%"struct.std::multiplies"*, i64* dereferenceable(8), i64* dereferenceable(8)) #1 comdat align 2 {
  %4 = alloca %"struct.std::multiplies"*, align 8
; CHECK-MUL:  %[[ALLOC1:.*]] = alloca i64 addrspace(4)*
  %5 = alloca i64*, align 8
; CHECK-MUL:  %[[ALLOC2:.*]] = alloca i64 addrspace(4)*
  %6 = alloca i64*, align 8
  store %"struct.std::multiplies"* %0, %"struct.std::multiplies"** %4, align 8
; CHECK-MUL:  store i64 addrspace(4)* %{{.*}}, i64 addrspace(4)** %[[ALLOC1]], align 8
  store i64* %1, i64** %5, align 8
; CHECK-MUL:  store i64 addrspace(4)* %{{.*}}, i64 addrspace(4)** %[[ALLOC2]], align 8
  store i64* %2, i64** %6, align 8
  %7 = load %"struct.std::multiplies"*, %"struct.std::multiplies"** %4, align 8
; CHECK-MUL:  %[[LOAD1:.*]] = load i64 addrspace(4)*, i64 addrspace(4)** %[[ALLOC1]]
  %8 = load i64*, i64** %5, align 8
; CHECK-MUL:  %[[NEXT_LOAD1:.*]] = load i64, i64 addrspace(4)* %[[LOAD1]]
  %9 = load i64, i64* %8, align 8
; CHECK-MUL:  %[[LOAD2:.*]] = load i64 addrspace(4)*, i64 addrspace(4)** %[[ALLOC2]]
  %10 = load i64*, i64** %6, align 8
; CHECK-MUL:  %[[NEXT_LOAD2:.*]] = load i64, i64 addrspace(4)* %[[LOAD2]]
  %11 = load i64, i64* %10, align 8
; CHECK-MUL:  %[[MUL:.*]] = mul nsw i64 %[[NEXT_LOAD1]], %[[NEXT_LOAD2]]
  %12 = mul nsw i64 %9, %11
; CHECK-MUL:  ret i64 %[[MUL]]
  ret i64 %12
}

attributes #0 = { noinline "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 8.0.0"}
