; RUN: split-file %s %t
; RUN: opt -passes=pgo-instr-gen,verify -pgo-instrument-entry -S %t/loop.ll | FileCheck %s --check-prefix=GEN --implicit-check-not='call void @llvm.instrprof.increment'
; RUN: %python %t/raw.py > %t/loop.raw
; RUN: llvm-profdata merge %t/loop.raw -o %t/loop.profdata
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/loop.profdata -S %t/loop.ll -o %t/use.ll
; RUN: FileCheck %s --check-prefixes=COUNTS,WAVE < %t/use.ll
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/loop.profdata -pgo-wave-metadata=false -S %t/loop.ll | FileCheck %s --check-prefix=COUNTS --implicit-check-not=wave.profile
; RUN: opt -passes=pgo-instr-gen,verify -pgo-instrument-entry -S %t/eh.ll | FileCheck %s --check-prefix=EH --implicit-check-not='call void @llvm.instrprof.increment'

; Dense instrumentation preserves the entry and split backedge counters,
; then appends the header and exit measurements.
; GEN: @__llvm_profile_raw_version = {{.*}}constant i64 378302368699121676
; GEN-LABEL: define void @loop(
; GEN: entry:
; GEN-NEXT: call void @llvm.instrprof.increment({{.*}}i64 [[HASH:784007056844089447]], i32 4, i32 0)
; GEN-NEXT: br label %header
; GEN: header:
; GEN-NEXT: %i = phi i32 [ 0, %entry ], [ %next, %header.header_crit_edge ]
; GEN-NEXT: call void @llvm.instrprof.increment.step({{.*}}i64 [[HASH]], i32 4, i32 2, i64 0)
; GEN: header.header_crit_edge:
; GEN-NEXT: call void @llvm.instrprof.increment({{.*}}i64 [[HASH]], i32 4, i32 1)
; GEN-NEXT: br label %header
; GEN: exit:
; GEN-NEXT: call void @llvm.instrprof.increment.step({{.*}}i64 [[HASH]], i32 4, i32 3, i64 0)
; GEN-NEXT: ret void

; Profile use reconstructs counter placement from the recorded flags. The
; header has four times the entry wave count; sparse data cannot supply it.
; COUNTS-LABEL: define void @loop(
; COUNTS-SAME: !prof ![[ENTRY_COUNT:[0-9]+]]
; WAVE-SAME: !wave.profile ![[WAVES:[0-9]+]]
; WAVE: entry:
; WAVE-NEXT: br label %header, {{.*}}!wave.profile.block ![[ENTRY:[0-9]+]]
; COUNTS: br i1 %done, label %exit, label %header.header_crit_edge, !prof ![[WEIGHTS:[0-9]+]]
; WAVE-SAME: !wave.profile.block ![[HEADER:[0-9]+]]
; WAVE: header.header_crit_edge:
; WAVE-NEXT: br label %header, {{.*}}!wave.profile.block ![[BACKEDGE:[0-9]+]]
; WAVE: exit:
; WAVE-NEXT: ret void, !wave.profile.block ![[EXIT:[0-9]+]]
; COUNTS-DAG: ![[ENTRY_COUNT]] = !{!"function_entry_count", i64 192}
; COUNTS-DAG: ![[WEIGHTS]] = !{!"branch_weights", i32 192, i32 576}
; WAVE-DAG: ![[WAVES]] = distinct !{i64 2, i64 [[FID:-?[0-9]+]], i64 3, i64 12, i64 9, i64 3}
; WAVE-DAG: ![[ENTRY]] = !{i64 2, i64 [[FID]], i64 0, i64 1, i64 1}
; WAVE-DAG: ![[HEADER]] = !{i64 2, i64 [[FID]], i64 1, i64 1, i64 3, i64 2}
; WAVE-DAG: ![[BACKEDGE]] = !{i64 2, i64 [[FID]], i64 2, i64 1, i64 1}
; WAVE-DAG: ![[EXIT]] = !{i64 2, i64 [[FID]], i64 3, i64 1}

; A catchswitch has no legal insertion point. Skip dispatch and append only
; exit; the sparse catch counter must remain after its catchpad.
; EH-LABEL: define void @eh(
; EH: entry:
; EH-NEXT: call void @llvm.instrprof.increment({{.*}}i64 [[EH_HASH:146835646621254984]], i32 3, i32 0)
; EH: dispatch:
; EH-NEXT: %cs = catchswitch within none [label %catch] unwind to caller
; EH: catch:
; EH-NEXT: %cp = catchpad within %cs [ptr null, i32 64, ptr null]
; EH-NEXT: call void @llvm.instrprof.increment({{.*}}i64 [[EH_HASH]], i32 3, i32 1)
; EH-NEXT: catchret from %cp to label %exit
; EH: exit:
; EH-NEXT: call void @llvm.instrprof.increment.step({{.*}}i64 [[EH_HASH]], i32 3, i32 2, i64 0)
; EH-NEXT: ret void

;--- loop.ll
target triple = "amdgcn-amd-amdhsa"

define void @loop(ptr %p, i32 %n) {
entry:
  br label %header
header:
  %i = phi i32 [0, %entry], [%next, %header]
  store volatile i32 %i, ptr %p
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %n
  br i1 %done, label %exit, label %header
exit:
  ret void
}

;--- eh.ll
target triple = "amdgcn-amd-amdhsa"

define void @eh() personality ptr @__CxxFrameHandler3 {
entry:
  invoke void @may_throw() to label %exit unwind label %dispatch
dispatch:
  %cs = catchswitch within none [label %catch] unwind to caller
catch:
  %cp = catchpad within %cs [ptr null, i32 64, ptr null]
  catchret from %cp to label %exit
exit:
  ret void
}

declare void @may_throw()
declare i32 @__CxxFrameHandler3(...)

;--- raw.py
import hashlib
import struct
import sys

# Three full waves, each taking four loop iterations. Counter order is
# entry, backedge, header, exit, as checked in the generation above.
name = b"loop"
name_ref = int.from_bytes(hashlib.md5(name).digest()[:8], "little")
lanes = [192, 576, 0, 0]
waves = [3, 9, 12, 3]
version = 12 | (1 << 54) | (1 << 56) | (1 << 58)
names = bytes([len(name), 0]) + name
# Raw v12: 80-byte data record, eight i64 lane/wave counters, four uniform counters.
counter_delta = 80
uniform_delta = counter_delta + 8 * (len(lanes) + len(waves))
names_delta = uniform_delta + 8 * len(lanes)
header = [0xff6c70726f667281, version, 0, 1, 0, 8, 0,
          0, 0, 4, 0, uniform_delta, len(names), counter_delta,
          uniform_delta, names_delta, 0, 0, 2]
record = struct.pack("<7QI4HII4x", name_ref, 784007056844089447, counter_delta,
                     uniform_delta, 0, 0, 0, 8, 0, 0, 0, 64, 0, 4)
counts = lanes + waves + lanes
sys.stdout.buffer.write(struct.pack("<19Q", *header) + record +
                        struct.pack("<12Q", *counts) +
                        names + bytes((-len(names)) % 8))
