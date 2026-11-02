; RUN: split-file %s %t
; RUN: llvm-profdata merge %t/original.proftext -o %t/original.profdata
; RUN: llvm-profdata merge %t/outlined.proftext -o %t/outlined.profdata
;
; Profiles must match the CFG that was instrumented. Matching profiles work
; before and after entry/body separation. A backend receiving already-profiled
; IR from an older producer must retain those annotations when it separates the
; entry. Conversely, feeding an old-shape profile to a new outlined frontend
; must diagnose the stale data, rather than attach it to the wrong function.
;
; RUN: opt -verify-each -passes='function(simplifycfg,sroa,early-cse),pgo-instr-use' -pgo-test-profile-file=%t/original.profdata -pgo-warn-missing-function -S %t/input.ll 2>&1 | FileCheck %s --check-prefix=ORIGINAL --implicit-check-not=warning:
; RUN: opt -verify-each -passes='coro-early,function(simplifycfg,sroa,early-cse),pgo-instr-use' -pgo-test-profile-file=%t/outlined.profdata -pgo-warn-missing-function -S %t/input.ll 2>&1 | FileCheck %s --check-prefix=BODY --implicit-check-not=warning:
; RUN: opt -verify-each -passes='function(simplifycfg,sroa,early-cse),pgo-instr-use,coro-early' -pgo-test-profile-file=%t/original.profdata -pgo-warn-missing-function -S %t/input.ll 2>&1 | FileCheck %s --check-prefix=BODY --implicit-check-not=warning:
; RUN: opt -verify-each -passes='coro-early,function(simplifycfg,sroa,early-cse),pgo-instr-use' -pgo-test-profile-file=%t/original.profdata -pgo-warn-missing-function -disable-output %t/input.ll 2>&1 | FileCheck %s --check-prefix=STALE
;
; ORIGINAL-LABEL: define { ptr, i32 } @yield_once(
; ORIGINAL-SAME: !prof ![[ENTRY:[0-9]+]]
; ORIGINAL: ![[ENTRY]] = !{!"function_entry_count", i64 1000}
;
; BODY-LABEL: define internal { ptr, i32 } @yield_once.coro.body(
; BODY-SAME: !prof ![[ENTRY:[0-9]+]]
; BODY: ![[ENTRY]] = !{!"function_entry_count", i64 1000}
;
; STALE: function control flow change detected (hash mismatch) yield_once
; STALE: no profile data available for function yield_once.coro.body

;--- input.ll
source_filename = "retcon-ramp-pgo.ll"

define { ptr, i32 } @yield_once(ptr %buffer) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id.retcon.once(i32 16, i32 8, ptr %buffer,
      ptr @prototype, ptr @malloc, ptr @free)
  %handle = call ptr @llvm.coro.begin(token %id, ptr null)
  %unwind = call i1 (...) @llvm.coro.suspend.retcon.i1(i32 7)
  br i1 %unwind, label %cleanup, label %cleanup

cleanup:
  call void @llvm.coro.end(ptr %handle, i1 false, token none)
  unreachable
}

declare token @llvm.coro.id.retcon.once(i32, i32, ptr, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.suspend.retcon.i1(...)
declare void @llvm.coro.end(ptr, i1, token)
declare void @prototype(ptr, i1)
declare ptr @malloc(i64)
declare void @free(ptr)

;--- original.proftext
:ir
yield_once
# Func Hash:
742261418966908927
# Num Counters:
1
1000

;--- outlined.proftext
:ir
yield_once
# Func Hash:
170957022131388415
# Num Counters:
1
1000
# Num Value Kinds:
1
# ValueKind = IPVK_IndirectCallTarget:
0
# NumValueSites:
1
# NumValueData for site 0:
0

retcon-ramp-pgo.ll;yield_once.coro.body
# Func Hash:
742261418966908927
# Num Counters:
1
1000
