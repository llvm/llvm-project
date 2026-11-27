; RUN: split-file %s %t
; RUN: not llvm-as < %t/masked-store.ll 2>&1 | FileCheck %s --check-prefix=MASKED-STORE
; RUN: not llvm-as < %t/masked-store-zero.ll 2>&1 | FileCheck %s --check-prefix=MASKED-STORE-ZERO
; RUN: not llvm-as < %t/masked-load.ll 2>&1 | FileCheck %s --check-prefix=MASKED-LOAD
; RUN: not llvm-as < %t/masked-load-zero.ll 2>&1 | FileCheck %s --check-prefix=MASKED-LOAD-ZERO
; RUN: not llvm-as < %t/masked-scatter.ll 2>&1 | FileCheck %s --check-prefix=MASKED-SCATTER
; RUN: not llvm-as < %t/masked-gather.ll 2>&1 | FileCheck %s --check-prefix=MASKED-GATHER

;--- masked-store.ll
; MASKED-STORE: LLVM ERROR: Invalid alignment argument
define void @masked_store(ptr %ptr, <2 x i1> %mask, <2 x double> %val) {
  call void @llvm.masked.store.v2f64.p0(<2 x double> %val, ptr %ptr, i32 3, <2 x i1> %mask)
  ret void
}

;--- masked-store-zero.ll
; MASKED-STORE-ZERO: LLVM ERROR: Invalid zero alignment argument
define void @masked_store_zero(ptr %ptr, <2 x i1> %mask, <2 x double> %val) {
  call void @llvm.masked.store.v2f64.p0(<2 x double> %val, ptr %ptr, i32 0, <2 x i1> %mask)
  ret void
}

;--- masked-load.ll
; MASKED-LOAD: LLVM ERROR: Invalid alignment argument
define void @masked_load(ptr %ptr, <2 x i1> %mask, <2 x double> %val) {
  call <2 x double> @llvm.masked.load.v2f64.p0(ptr %ptr, i32 3, <2 x i1> %mask, <2 x double> %val)
  ret void
}

;--- masked-load-zero.ll
; MASKED-LOAD-ZERO: LLVM ERROR: Invalid zero alignment argument
define void @masked_load_zero(ptr %ptr, <2 x i1> %mask, <2 x double> %val) {
  call <2 x double> @llvm.masked.load.v2f64.p0(ptr %ptr, i32 0, <2 x i1> %mask, <2 x double> %val)
  ret void
}

;--- masked-scatter.ll
; MASKED-SCATTER: LLVM ERROR: Invalid alignment argument
define void @masked_scatter(<2 x ptr> %ptr, <2 x i1> %mask, <2 x double> %val) {
  call void @llvm.masked.scatter.v2f64.p0(<2 x double> %val, <2 x ptr> %ptr, i32 3, <2 x i1> %mask)
  ret void
}

;--- masked-gather.ll
; MASKED-GATHER: LLVM ERROR: Invalid alignment argument
define void @masked_gather(<2 x ptr> %ptr, <2 x i1> %mask, <2 x double> %val) {
  call <2 x double> @llvm.masked.gather.v2f64.p0(<2 x ptr> %ptr, i32 3, <2 x i1> %mask, <2 x double> %val)
  ret void
}
