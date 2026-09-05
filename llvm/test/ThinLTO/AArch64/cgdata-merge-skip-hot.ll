; Verify profile policies for global function merging in local and CGData modes.
;
; The policy is deliberately not serialized in CGData, so this does not change
; the CGData format. Producers and consumers should use the same policy for best
; profitability. A consumer of stale CGData still applies its current policy to
; each function, but stale candidates can affect parameter selection.

; RUN: rm -rf %t; split-file %s %t

; Local mode has no profile summary. The hot policy therefore excludes only
; functions with an explicit hot attribute. The executed policy additionally
; excludes functions with a nonzero entry count. Zero-count and unprofiled
; functions remain eligible under both policies.
; RUN: opt -mtriple=arm64-apple-darwin -S --passes=global-merge-func \
; RUN:   -global-merging-profile-policy=none %t/local.ll -o %t-local-none.ll
; RUN: FileCheck %s --check-prefix=LOCAL-NONE < %t-local-none.ll
; RUN: opt -mtriple=arm64-apple-darwin -S --passes=global-merge-func \
; RUN:   -global-merging-profile-policy=hot %t/local.ll -o %t-local-hot.ll
; RUN: FileCheck %s --check-prefix=LOCAL-HOT < %t-local-hot.ll
; RUN: not grep -E '@attr_hot_local_[12]\.Tgm' %t-local-hot.ll
; RUN: opt -mtriple=arm64-apple-darwin -S --passes=global-merge-func \
; RUN:   -global-merging-profile-policy=executed %t/local.ll \
; RUN:   -o %t-local-executed.ll
; RUN: FileCheck %s --check-prefix=LOCAL-EXECUTED < %t-local-executed.ll
; RUN: not grep -E '@(attr_hot|executed)_local_[12]\.Tgm' \
; RUN:   %t-local-executed.ll

; LOCAL-NONE-DAG: @attr_hot_local_1.Tgm
; LOCAL-NONE-DAG: @attr_hot_local_2.Tgm
; LOCAL-NONE-DAG: @executed_local_1.Tgm
; LOCAL-NONE-DAG: @executed_local_2.Tgm
; LOCAL-NONE-DAG: @zero_local_1.Tgm
; LOCAL-NONE-DAG: @zero_local_2.Tgm
; LOCAL-NONE-DAG: @unprofiled_local_1.Tgm
; LOCAL-NONE-DAG: @unprofiled_local_2.Tgm
; LOCAL-HOT-DAG: @executed_local_1.Tgm
; LOCAL-HOT-DAG: @executed_local_2.Tgm
; LOCAL-HOT-DAG: @zero_local_1.Tgm
; LOCAL-HOT-DAG: @zero_local_2.Tgm
; LOCAL-HOT-DAG: @unprofiled_local_1.Tgm
; LOCAL-HOT-DAG: @unprofiled_local_2.Tgm
; LOCAL-EXECUTED-DAG: @zero_local_1.Tgm
; LOCAL-EXECUTED-DAG: @zero_local_2.Tgm
; LOCAL-EXECUTED-DAG: @unprofiled_local_1.Tgm
; LOCAL-EXECUTED-DAG: @unprofiled_local_2.Tgm

; Build ThinLTO inputs with profile summaries so the hot policy can classify
; profile-hot functions.
; RUN: opt -module-summary -module-hash %t/foo.ll -o %t-foo.bc
; RUN: opt -module-summary -module-hash %t/bar.ll -o %t-bar.bc

; Produce CGData without profile filtering.
; RUN: llvm-lto2 run -enable-global-merge-func=true \
; RUN:   -codegen-data-generate=true \
; RUN:   %t-foo.bc %t-bar.bc -o %t-default-write \
; RUN:   -r %t-foo.bc,_profile_hot_1,px -r %t-bar.bc,_profile_hot_2,px \
; RUN:   -r %t-foo.bc,_attr_hot_1,px -r %t-bar.bc,_attr_hot_2,px \
; RUN:   -r %t-bar.bc,_mixed_hot,px -r %t-bar.bc,_mixed_cold_1,px \
; RUN:   -r %t-bar.bc,_mixed_cold_2,px \
; RUN:   -r %t-foo.bc,_cold_1,px -r %t-bar.bc,_cold_2,px \
; RUN:   -r %t-foo.bc,_zero_1,px -r %t-bar.bc,_zero_2,px \
; RUN:   -r %t-foo.bc,_unprofiled_1,px -r %t-bar.bc,_unprofiled_2,px \
; RUN:   -r %t-foo.bc,_g1,l -r %t-bar.bc,_g2,l -r %t-bar.bc,_g3,l \
; RUN:   -r %t-bar.bc,_g4,l
; RUN: llvm-cgdata --merge -o %t-default.cgdata \
; RUN:   %t-default-write.1 %t-default-write.2
; RUN: llvm-cgdata --convert %t-default.cgdata -o %t-default.yaml
; RUN: FileCheck %s --check-prefix=DEFAULT-CGDATA < %t-default.yaml

; DEFAULT-CGDATA-DAG: FunctionName: mixed_hot
; DEFAULT-CGDATA-DAG: FunctionName: mixed_cold_1
; DEFAULT-CGDATA-DAG: FunctionName: mixed_cold_2
; DEFAULT-CGDATA-DAG: FunctionName: zero_1
; DEFAULT-CGDATA-DAG: FunctionName: zero_2

; Default behavior remains unchanged and merges hot functions.
; RUN: llvm-lto2 run -enable-global-merge-func=true \
; RUN:   -codegen-data-use-path=%t-default.cgdata \
; RUN:   %t-foo.bc %t-bar.bc -o %t-default-read \
; RUN:   -r %t-foo.bc,_profile_hot_1,px -r %t-bar.bc,_profile_hot_2,px \
; RUN:   -r %t-foo.bc,_attr_hot_1,px -r %t-bar.bc,_attr_hot_2,px \
; RUN:   -r %t-bar.bc,_mixed_hot,px -r %t-bar.bc,_mixed_cold_1,px \
; RUN:   -r %t-bar.bc,_mixed_cold_2,px \
; RUN:   -r %t-foo.bc,_cold_1,px -r %t-bar.bc,_cold_2,px \
; RUN:   -r %t-foo.bc,_zero_1,px -r %t-bar.bc,_zero_2,px \
; RUN:   -r %t-foo.bc,_unprofiled_1,px -r %t-bar.bc,_unprofiled_2,px \
; RUN:   -r %t-foo.bc,_g1,l -r %t-bar.bc,_g2,l -r %t-bar.bc,_g3,l \
; RUN:   -r %t-bar.bc,_g4,l
; RUN: llvm-nm %t-default-read.1 > %t-default-foo.nm
; RUN: llvm-nm %t-default-read.2 > %t-default-bar.nm
; RUN: FileCheck %s --check-prefix=DEFAULT-FOO < %t-default-foo.nm
; RUN: FileCheck %s --check-prefix=DEFAULT-BAR < %t-default-bar.nm

; DEFAULT-FOO-DAG: _profile_hot_1.Tgm
; DEFAULT-FOO-DAG: _attr_hot_1.Tgm
; DEFAULT-FOO-DAG: _zero_1.Tgm
; DEFAULT-BAR-DAG: _profile_hot_2.Tgm
; DEFAULT-BAR-DAG: _attr_hot_2.Tgm
; DEFAULT-BAR-DAG: _mixed_hot.Tgm
; DEFAULT-BAR-DAG: _zero_2.Tgm

; Produce fresh CGData using the hot policy. The same-module group contains one
; hot function and two cold functions: only the hot member is filtered, while
; the two cold members remain a profitable finalized candidate.
; RUN: llvm-lto2 run -enable-global-merge-func=true \
; RUN:   -global-merging-profile-policy=hot -codegen-data-generate=true \
; RUN:   %t-foo.bc %t-bar.bc -o %t-hot-write \
; RUN:   -r %t-foo.bc,_profile_hot_1,px -r %t-bar.bc,_profile_hot_2,px \
; RUN:   -r %t-foo.bc,_attr_hot_1,px -r %t-bar.bc,_attr_hot_2,px \
; RUN:   -r %t-bar.bc,_mixed_hot,px -r %t-bar.bc,_mixed_cold_1,px \
; RUN:   -r %t-bar.bc,_mixed_cold_2,px \
; RUN:   -r %t-foo.bc,_cold_1,px -r %t-bar.bc,_cold_2,px \
; RUN:   -r %t-foo.bc,_zero_1,px -r %t-bar.bc,_zero_2,px \
; RUN:   -r %t-foo.bc,_unprofiled_1,px -r %t-bar.bc,_unprofiled_2,px \
; RUN:   -r %t-foo.bc,_g1,l -r %t-bar.bc,_g2,l -r %t-bar.bc,_g3,l \
; RUN:   -r %t-bar.bc,_g4,l
; RUN: llvm-cgdata --merge -o %t-hot.cgdata %t-hot-write.1 %t-hot-write.2
; RUN: llvm-cgdata --convert %t-hot.cgdata -o %t-hot.yaml
; RUN: FileCheck %s --check-prefix=HOT-CGDATA < %t-hot.yaml
; RUN: not grep -E 'profile_hot|attr_hot|mixed_hot' %t-hot.yaml

; HOT-CGDATA-DAG: FunctionName: cold_1
; HOT-CGDATA-DAG: FunctionName: cold_2
; HOT-CGDATA-DAG: FunctionName: zero_1
; HOT-CGDATA-DAG: FunctionName: zero_2
; HOT-CGDATA-DAG: FunctionName: unprofiled_1
; HOT-CGDATA-DAG: FunctionName: unprofiled_2
; HOT-CGDATA-DAG: FunctionName: mixed_cold_1
; HOT-CGDATA-DAG: FunctionName: mixed_cold_2

; Consume fresh CGData with the same hot policy.
; RUN: llvm-lto2 run -enable-global-merge-func=true \
; RUN:   -global-merging-profile-policy=hot \
; RUN:   -codegen-data-use-path=%t-hot.cgdata \
; RUN:   %t-foo.bc %t-bar.bc -o %t-hot-read \
; RUN:   -r %t-foo.bc,_profile_hot_1,px -r %t-bar.bc,_profile_hot_2,px \
; RUN:   -r %t-foo.bc,_attr_hot_1,px -r %t-bar.bc,_attr_hot_2,px \
; RUN:   -r %t-bar.bc,_mixed_hot,px -r %t-bar.bc,_mixed_cold_1,px \
; RUN:   -r %t-bar.bc,_mixed_cold_2,px \
; RUN:   -r %t-foo.bc,_cold_1,px -r %t-bar.bc,_cold_2,px \
; RUN:   -r %t-foo.bc,_zero_1,px -r %t-bar.bc,_zero_2,px \
; RUN:   -r %t-foo.bc,_unprofiled_1,px -r %t-bar.bc,_unprofiled_2,px \
; RUN:   -r %t-foo.bc,_g1,l -r %t-bar.bc,_g2,l -r %t-bar.bc,_g3,l \
; RUN:   -r %t-bar.bc,_g4,l
; RUN: llvm-nm %t-hot-read.1 > %t-hot-foo.nm
; RUN: llvm-nm %t-hot-read.2 > %t-hot-bar.nm
; RUN: FileCheck %s --check-prefix=HOT-FOO < %t-hot-foo.nm
; RUN: FileCheck %s --check-prefix=HOT-BAR < %t-hot-bar.nm
; RUN: not grep -E '_profile_hot_1\.Tgm|_attr_hot_1\.Tgm' %t-hot-foo.nm
; RUN: not grep -E '_profile_hot_2\.Tgm|_attr_hot_2\.Tgm|_mixed_hot\.Tgm' \
; RUN:   %t-hot-bar.nm

; HOT-FOO-DAG: _cold_1.Tgm
; HOT-FOO-DAG: _zero_1.Tgm
; HOT-FOO-DAG: _unprofiled_1.Tgm
; HOT-BAR-DAG: _cold_2.Tgm
; HOT-BAR-DAG: _zero_2.Tgm
; HOT-BAR-DAG: _unprofiled_2.Tgm
; HOT-BAR-DAG: _mixed_cold_1.Tgm
; HOT-BAR-DAG: _mixed_cold_2.Tgm

; Consume stale unfiltered CGData with the hot policy. The consumer still skips
; only each currently hot function. In particular, it does not poison the hash
; shared by the two cold peers. Matching production and consumption policies is
; nevertheless preferred because stale records can change profitability and
; parameter selection.
; RUN: llvm-lto2 run -enable-global-merge-func=true \
; RUN:   -global-merging-profile-policy=hot \
; RUN:   -codegen-data-use-path=%t-default.cgdata \
; RUN:   %t-foo.bc %t-bar.bc -o %t-stale-read \
; RUN:   -r %t-foo.bc,_profile_hot_1,px -r %t-bar.bc,_profile_hot_2,px \
; RUN:   -r %t-foo.bc,_attr_hot_1,px -r %t-bar.bc,_attr_hot_2,px \
; RUN:   -r %t-bar.bc,_mixed_hot,px -r %t-bar.bc,_mixed_cold_1,px \
; RUN:   -r %t-bar.bc,_mixed_cold_2,px \
; RUN:   -r %t-foo.bc,_cold_1,px -r %t-bar.bc,_cold_2,px \
; RUN:   -r %t-foo.bc,_zero_1,px -r %t-bar.bc,_zero_2,px \
; RUN:   -r %t-foo.bc,_unprofiled_1,px -r %t-bar.bc,_unprofiled_2,px \
; RUN:   -r %t-foo.bc,_g1,l -r %t-bar.bc,_g2,l -r %t-bar.bc,_g3,l \
; RUN:   -r %t-bar.bc,_g4,l
; RUN: llvm-nm %t-stale-read.1 > %t-stale-foo.nm
; RUN: llvm-nm %t-stale-read.2 > %t-stale-bar.nm
; RUN: FileCheck %s --check-prefix=STALE-FOO < %t-stale-foo.nm
; RUN: FileCheck %s --check-prefix=STALE-BAR < %t-stale-bar.nm
; RUN: not grep -E '_profile_hot_1\.Tgm|_attr_hot_1\.Tgm' %t-stale-foo.nm
; RUN: not grep -E '_profile_hot_2\.Tgm|_attr_hot_2\.Tgm|_mixed_hot\.Tgm' \
; RUN:   %t-stale-bar.nm

; STALE-FOO-DAG: _cold_1.Tgm
; STALE-FOO-DAG: _zero_1.Tgm
; STALE-FOO-DAG: _unprofiled_1.Tgm
; STALE-BAR-DAG: _cold_2.Tgm
; STALE-BAR-DAG: _zero_2.Tgm
; STALE-BAR-DAG: _unprofiled_2.Tgm
; STALE-BAR-DAG: _mixed_cold_1.Tgm
; STALE-BAR-DAG: _mixed_cold_2.Tgm

; Produce CGData using the executed policy. Explicitly hot and every function
; with a nonzero entry count are absent, while zero-count and unprofiled pairs
; remain in the finalized map.
; RUN: llvm-lto2 run -enable-global-merge-func=true \
; RUN:   -global-merging-profile-policy=executed -codegen-data-generate=true \
; RUN:   %t-foo.bc %t-bar.bc -o %t-executed-write \
; RUN:   -r %t-foo.bc,_profile_hot_1,px -r %t-bar.bc,_profile_hot_2,px \
; RUN:   -r %t-foo.bc,_attr_hot_1,px -r %t-bar.bc,_attr_hot_2,px \
; RUN:   -r %t-bar.bc,_mixed_hot,px -r %t-bar.bc,_mixed_cold_1,px \
; RUN:   -r %t-bar.bc,_mixed_cold_2,px \
; RUN:   -r %t-foo.bc,_cold_1,px -r %t-bar.bc,_cold_2,px \
; RUN:   -r %t-foo.bc,_zero_1,px -r %t-bar.bc,_zero_2,px \
; RUN:   -r %t-foo.bc,_unprofiled_1,px -r %t-bar.bc,_unprofiled_2,px \
; RUN:   -r %t-foo.bc,_g1,l -r %t-bar.bc,_g2,l -r %t-bar.bc,_g3,l \
; RUN:   -r %t-bar.bc,_g4,l
; RUN: llvm-cgdata --merge -o %t-executed.cgdata \
; RUN:   %t-executed-write.1 %t-executed-write.2
; RUN: llvm-cgdata --convert %t-executed.cgdata -o %t-executed.yaml
; RUN: FileCheck %s --check-prefix=EXECUTED-CGDATA < %t-executed.yaml
; RUN: not grep -E 'profile_hot|attr_hot|mixed_|FunctionName: cold_[12]' \
; RUN:   %t-executed.yaml

; EXECUTED-CGDATA-DAG: FunctionName: zero_1
; EXECUTED-CGDATA-DAG: FunctionName: zero_2
; EXECUTED-CGDATA-DAG: FunctionName: unprofiled_1
; EXECUTED-CGDATA-DAG: FunctionName: unprofiled_2

; Consume unfiltered CGData with the executed policy. The independent
; consumer-side guard prevents merged instances for every executed or explicit
; hot function, while zero-count and unprofiled groups remain profitable.
; RUN: llvm-lto2 run -enable-global-merge-func=true \
; RUN:   -global-merging-profile-policy=executed \
; RUN:   -codegen-data-use-path=%t-default.cgdata \
; RUN:   %t-foo.bc %t-bar.bc -o %t-executed-read \
; RUN:   -r %t-foo.bc,_profile_hot_1,px -r %t-bar.bc,_profile_hot_2,px \
; RUN:   -r %t-foo.bc,_attr_hot_1,px -r %t-bar.bc,_attr_hot_2,px \
; RUN:   -r %t-bar.bc,_mixed_hot,px -r %t-bar.bc,_mixed_cold_1,px \
; RUN:   -r %t-bar.bc,_mixed_cold_2,px \
; RUN:   -r %t-foo.bc,_cold_1,px -r %t-bar.bc,_cold_2,px \
; RUN:   -r %t-foo.bc,_zero_1,px -r %t-bar.bc,_zero_2,px \
; RUN:   -r %t-foo.bc,_unprofiled_1,px -r %t-bar.bc,_unprofiled_2,px \
; RUN:   -r %t-foo.bc,_g1,l -r %t-bar.bc,_g2,l -r %t-bar.bc,_g3,l \
; RUN:   -r %t-bar.bc,_g4,l
; RUN: llvm-nm %t-executed-read.1 > %t-executed-foo.nm
; RUN: llvm-nm %t-executed-read.2 > %t-executed-bar.nm
; RUN: FileCheck %s --check-prefix=EXECUTED-FOO < %t-executed-foo.nm
; RUN: FileCheck %s --check-prefix=EXECUTED-BAR < %t-executed-bar.nm
; RUN: not grep -E '_profile_hot_1\.Tgm|_attr_hot_1\.Tgm|_cold_1\.Tgm' \
; RUN:   %t-executed-foo.nm
; RUN: not grep -E '_profile_hot_2\.Tgm|_attr_hot_2\.Tgm|_mixed_(hot|cold_[12])\.Tgm|_cold_2\.Tgm' \
; RUN:   %t-executed-bar.nm

; EXECUTED-FOO-DAG: _zero_1.Tgm
; EXECUTED-FOO-DAG: _unprofiled_1.Tgm
; EXECUTED-BAR-DAG: _zero_2.Tgm
; EXECUTED-BAR-DAG: _unprofiled_2.Tgm

;--- local.ll
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-unknown-ios12.0.0"

@local_g1 = external global i32
@local_g2 = external global i32

define i32 @attr_hot_local_1() #0 {
  %value = load volatile i32, ptr @local_g1
  %a = add i32 %value, 1
  %b = mul i32 %a, 2
  %c = xor i32 %b, 3
  %d = sub i32 %c, 4
  %result = or i32 %d, 5
  ret i32 %result
}

define i32 @attr_hot_local_2() #0 {
  %value = load volatile i32, ptr @local_g2
  %a = add i32 %value, 1
  %b = mul i32 %a, 2
  %c = xor i32 %b, 3
  %d = sub i32 %c, 4
  %result = or i32 %d, 5
  ret i32 %result
}

define i32 @executed_local_1() !prof !0 {
  %value = load volatile i32, ptr @local_g1
  %a = mul i32 %value, 1
  %b = add i32 %a, 2
  %c = xor i32 %b, 3
  %d = sub i32 %c, 4
  %result = or i32 %d, 5
  ret i32 %result
}

define i32 @executed_local_2() !prof !0 {
  %value = load volatile i32, ptr @local_g2
  %a = mul i32 %value, 1
  %b = add i32 %a, 2
  %c = xor i32 %b, 3
  %d = sub i32 %c, 4
  %result = or i32 %d, 5
  ret i32 %result
}

define i32 @zero_local_1() !prof !1 {
  %value = load volatile i32, ptr @local_g1
  %a = sub i32 %value, 1
  %b = mul i32 %a, 2
  %c = xor i32 %b, 3
  %d = add i32 %c, 4
  %result = or i32 %d, 5
  ret i32 %result
}

define i32 @zero_local_2() !prof !1 {
  %value = load volatile i32, ptr @local_g2
  %a = sub i32 %value, 1
  %b = mul i32 %a, 2
  %c = xor i32 %b, 3
  %d = add i32 %c, 4
  %result = or i32 %d, 5
  ret i32 %result
}

define i32 @unprofiled_local_1() {
  %v0 = load volatile i32, ptr @local_g1
  %v1 = load volatile i32, ptr @local_g1
  %v2 = load volatile i32, ptr @local_g1
  %v3 = load volatile i32, ptr @local_g1
  %a = add i32 %v0, %v1
  %b = add i32 %v2, %v3
  %result = xor i32 %a, %b
  ret i32 %result
}

define i32 @unprofiled_local_2() {
  %v0 = load volatile i32, ptr @local_g2
  %v1 = load volatile i32, ptr @local_g2
  %v2 = load volatile i32, ptr @local_g2
  %v3 = load volatile i32, ptr @local_g2
  %a = add i32 %v0, %v1
  %b = add i32 %v2, %v3
  %result = xor i32 %a, %b
  ret i32 %result
}

attributes #0 = { hot }

!0 = !{!"function_entry_count", i64 42}
!1 = !{!"function_entry_count", i64 0}

;--- foo.ll
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-unknown-ios12.0.0"

@g1 = external global i32

define i32 @profile_hot_1() !prof !20 {
  %value = load volatile i32, ptr @g1
  %a = add i32 %value, 1
  %b = mul i32 %a, 2
  %c = xor i32 %b, 3
  %d = sub i32 %c, 4
  %result = or i32 %d, 5
  ret i32 %result
}

define i32 @attr_hot_1() #0 {
  %value = load volatile i32, ptr @g1
  %a = mul i32 %value, 1
  %b = add i32 %a, 2
  %c = xor i32 %b, 3
  %d = sub i32 %c, 4
  %result = or i32 %d, 5
  ret i32 %result
}

define i32 @cold_1() !prof !21 {
  %value = load volatile i32, ptr @g1
  %a = sub i32 %value, 1
  %b = mul i32 %a, 2
  %c = xor i32 %b, 3
  %d = add i32 %c, 4
  %result = or i32 %d, 5
  ret i32 %result
}

define i32 @zero_1() !prof !22 {
  %value = load volatile i32, ptr @g1
  %a = xor i32 %value, 1
  %b = add i32 %a, 2
  %c = mul i32 %b, 3
  %d = sub i32 %c, 4
  %result = or i32 %d, 5
  ret i32 %result
}

define i32 @unprofiled_1() {
  %v0 = load volatile i32, ptr @g1
  %v1 = load volatile i32, ptr @g1
  %v2 = load volatile i32, ptr @g1
  %v3 = load volatile i32, ptr @g1
  %a = add i32 %v0, %v1
  %b = add i32 %v2, %v3
  %result = xor i32 %a, %b
  ret i32 %result
}

attributes #0 = { hot }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 10000}
!4 = !{!"MaxCount", i64 10}
!5 = !{!"MaxInternalCount", i64 1}
!6 = !{!"MaxFunctionCount", i64 1000}
!7 = !{!"NumCounts", i64 3}
!8 = !{!"NumFunctions", i64 3}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13}
!11 = !{i32 10000, i64 100, i32 1}
!12 = !{i32 999000, i64 100, i32 1}
!13 = !{i32 999999, i64 1, i32 2}
!20 = !{!"function_entry_count", i64 400}
!21 = !{!"function_entry_count", i64 1}
!22 = !{!"function_entry_count", i64 0}

;--- bar.ll
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-unknown-ios12.0.0"

@g2 = external global i32
@g3 = external global i32
@g4 = external global i32

define i32 @profile_hot_2() !prof !20 {
  %value = load volatile i32, ptr @g2
  %a = add i32 %value, 1
  %b = mul i32 %a, 2
  %c = xor i32 %b, 3
  %d = sub i32 %c, 4
  %result = or i32 %d, 5
  ret i32 %result
}

define i32 @attr_hot_2() #0 {
  %value = load volatile i32, ptr @g2
  %a = mul i32 %value, 1
  %b = add i32 %a, 2
  %c = xor i32 %b, 3
  %d = sub i32 %c, 4
  %result = or i32 %d, 5
  ret i32 %result
}

define i32 @mixed_hot() #1 !prof !20 {
  %value = load volatile i32, ptr @g2
  %a = shl i32 %value, 1
  %b = mul i32 %a, 2
  %c = xor i32 %b, 3
  %d = sub i32 %c, 4
  %result = or i32 %d, 5
  ret i32 %result
}

define i32 @mixed_cold_1() #1 !prof !21 {
  %value = load volatile i32, ptr @g3
  %a = shl i32 %value, 1
  %b = mul i32 %a, 2
  %c = xor i32 %b, 3
  %d = sub i32 %c, 4
  %result = or i32 %d, 5
  ret i32 %result
}

define i32 @mixed_cold_2() #1 !prof !21 {
  %value = load volatile i32, ptr @g4
  %a = shl i32 %value, 1
  %b = mul i32 %a, 2
  %c = xor i32 %b, 3
  %d = sub i32 %c, 4
  %result = or i32 %d, 5
  ret i32 %result
}

define i32 @cold_2() !prof !21 {
  %value = load volatile i32, ptr @g2
  %a = sub i32 %value, 1
  %b = mul i32 %a, 2
  %c = xor i32 %b, 3
  %d = add i32 %c, 4
  %result = or i32 %d, 5
  ret i32 %result
}

define i32 @zero_2() !prof !22 {
  %value = load volatile i32, ptr @g2
  %a = xor i32 %value, 1
  %b = add i32 %a, 2
  %c = mul i32 %b, 3
  %d = sub i32 %c, 4
  %result = or i32 %d, 5
  ret i32 %result
}

define i32 @unprofiled_2() {
  %v0 = load volatile i32, ptr @g2
  %v1 = load volatile i32, ptr @g2
  %v2 = load volatile i32, ptr @g2
  %v3 = load volatile i32, ptr @g2
  %a = add i32 %v0, %v1
  %b = add i32 %v2, %v3
  %result = xor i32 %a, %b
  ret i32 %result
}

attributes #0 = { hot }
attributes #1 = { noinline optnone }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 10000}
!4 = !{!"MaxCount", i64 10}
!5 = !{!"MaxInternalCount", i64 1}
!6 = !{!"MaxFunctionCount", i64 1000}
!7 = !{!"NumCounts", i64 3}
!8 = !{!"NumFunctions", i64 3}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13}
!11 = !{i32 10000, i64 100, i32 1}
!12 = !{i32 999000, i64 100, i32 1}
!13 = !{i32 999999, i64 1, i32 2}
!20 = !{!"function_entry_count", i64 400}
!21 = !{!"function_entry_count", i64 1}
!22 = !{!"function_entry_count", i64 0}
