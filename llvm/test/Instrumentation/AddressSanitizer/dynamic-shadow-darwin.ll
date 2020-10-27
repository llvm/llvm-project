; Test using dynamic shadow address on darwin
;
; RUN: opt -asan -asan-module -mtriple=arm64_32-apple-watchos --data-layout="e-m:o-p:32:32-i64:64-i128:128-n32:64-S128" -S < %s -enable-new-pm=0 | FileCheck %s --check-prefixes=CHECK,CHECK-DYNAMIC -DPTR_SIZE=32
; RUN: opt -passes='asan-pipeline' -mtriple=arm64_32-apple-watchos --data-layout="e-m:o-p:32:32-i64:64-i128:128-n32:64-S128" -S < %s | FileCheck %s --check-prefixes=CHECK,CHECK-DYNAMIC -DPTR_SIZE=32
; RUN: opt -asan -asan-module -mtriple=armv7k-apple-watchos --data-layout="e-m:o-p:32:32-Fi8-i64:64-a:0:32-n32-S128" -S < %s -enable-new-pm=0 | FileCheck %s --check-prefixes=CHECK,CHECK-DYNAMIC -DPTR_SIZE=32
; RUN: opt -passes='asan-pipeline' -mtriple=armv7k-apple-watchos --data-layout="e-m:o-p:32:32-Fi8-i64:64-a:0:32-n32-S128" -S < %s | FileCheck %s --check-prefixes=CHECK,CHECK-DYNAMIC -DPTR_SIZE=32
; RUN: opt -asan -asan-module -mtriple=arm64-apple-ios --data-layout="e-m:o-i64:64-i128:128-n32:64-S128" -S < %s -enable-new-pm=0 | FileCheck %s --check-prefixes=CHECK,CHECK-DYNAMIC -DPTR_SIZE=64
; RUN: opt -passes='asan-pipeline' -mtriple=arm64-apple-ios --data-layout="e-m:o-i64:64-i128:128-n32:64-S128" -S < %s | FileCheck %s --check-prefixes=CHECK,CHECK-DYNAMIC -DPTR_SIZE=64
; RUN: opt -asan -asan-module -mtriple=armv7s-apple-ios --data-layout="e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32" -S < %s -enable-new-pm=0 | FileCheck %s --check-prefixes=CHECK,CHECK-DYNAMIC -DPTR_SIZE=32
; RUN: opt -passes='asan-pipeline' -mtriple=armv7s-apple-ios --data-layout="e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32" -S < %s | FileCheck %s --check-prefixes=CHECK,CHECK-DYNAMIC -DPTR_SIZE=32
; RUN: opt -asan -asan-module -mtriple=i386-apple-watchos-simulator --data-layout="e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128" -S < %s -enable-new-pm=0 | FileCheck %s --check-prefixes=CHECK,CHECK-DYNAMIC -DPTR_SIZE=32
; RUN: opt -passes='asan-pipeline' -mtriple=i386-apple-watchos-simulator --data-layout="e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128" -S < %s | FileCheck %s --check-prefixes=CHECK,CHECK-DYNAMIC -DPTR_SIZE=32
; RUN: opt -asan -asan-module -mtriple=i386-apple-ios-simulator --data-layout="e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128" -S < %s -enable-new-pm=0 | FileCheck %s --check-prefixes=CHECK,CHECK-DYNAMIC -DPTR_SIZE=32
; RUN: opt -passes='asan-pipeline' -mtriple=i386-apple-ios-simulator --data-layout="e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128" -S < %s | FileCheck %s --check-prefixes=CHECK,CHECK-DYNAMIC -DPTR_SIZE=32
; RUN: opt -asan -asan-module -mtriple=x86_64-apple-ios-simulator --data-layout="e-m:o-i64:64-f80:128-n8:16:32:64-S128" -S < %s -enable-new-pm=0 | FileCheck %s --check-prefixes=CHECK,CHECK-DYNAMIC -DPTR_SIZE=64
; RUN: opt -passes='asan-pipeline' -mtriple=x86_64-apple-ios-simulator --data-layout="e-m:o-i64:64-f80:128-n8:16:32:64-S128" -S < %s | FileCheck %s --check-prefixes=CHECK,CHECK-DYNAMIC -DPTR_SIZE=64
;
; // macOS does not use dynamic shadow placement on x86_64
; RUN: opt -asan -asan-module -mtriple=x86_64-apple-macosx --data-layout="e-m:o-i64:64-f80:128-n8:16:32:64-S128" -S < %s -enable-new-pm=0 | FileCheck %s --check-prefixes=CHECK,CHECK-NONDYNAMIC -DPTR_SIZE=64
; RUN: opt -passes='asan-pipeline' -mtriple=x86_64-apple-macosx --data-layout="e-m:o-i64:64-f80:128-n8:16:32:64-S128" -S < %s | FileCheck %s --check-prefixes=CHECK,CHECK-NONDYNAMIC -DPTR_SIZE=64
; // macOS does use dynamic shadow placement on arm64
; RUN: opt -asan -asan-module -mtriple=arm64-apple-macosx --data-layout="e-m:o-i64:64-i128:128-n32:64-S128" -S < %s -enable-new-pm=0 | FileCheck %s --check-prefixes=CHECK,CHECK-DYNAMIC -DPTR_SIZE=64
; RUN: opt -passes='asan-pipeline' -mtriple=arm64-apple-macosx --data-layout="e-m:o-i64:64-i128:128-n32:64-S128" -S < %s | FileCheck %s --check-prefixes=CHECK,CHECK-DYNAMIC -DPTR_SIZE=64

define i32 @test_load(i32* %a) sanitize_address {
; First instrumentation in the function must be to load the dynamic shadow
; address into a local variable.
; CHECK-LABEL: @test_load
; CHECK: entry:
; CHECK-DYNAMIC-NEXT: %[[SHADOW:[^ ]*]] = load i[[PTR_SIZE]], i[[PTR_SIZE]]* @__asan_shadow_memory_dynamic_address
; CHECK-NONDYNAMIC-NOT: __asan_shadow_memory_dynamic_address

; Shadow address is loaded and added into the whole offset computation.
; CHECK-DYNAMIC: add i[[PTR_SIZE]] %{{.*}}, %[[SHADOW]]

entry:
  %tmp1 = load i32, i32* %a, align 4
  ret i32 %tmp1
}
